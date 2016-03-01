require 'torch'
require 'nn'
local hdf5 = require 'hdf5'
local pl = require('pl.import_into')()
torch.setdefaulttensortype('torch.FloatTensor')

-- Some definitions copied from the TensorFlow model
-- Epsilon value used in Batch Normalization
local std_epsilon = 0.0010000000475


local args = pl.lapp [[
  -i (string) folder with all h5 files dumped by `dump_filters.py`
  -b (string) backend to use: "nn"|"cunn"|"cudnn"
  -o (string) output torch binary file with the full model
]]

-- modules to be attached to a specific backend
local SpatialConvolution
local SpatialMaxPooling
local ReLU
local SoftMax
local SpatialBatchNormalization
if args.b == "nn" or args.b == "cunn" then
  SpatialConvolution = nn.SpatialConvolution
  SpatialMaxPooling = nn.SpatialMaxPooling
  ReLU = nn.ReLU
  SoftMax = nn.SoftMax
  SpatialBatchNormalization = nn.SpatialBatchNormalization
  if args.b == "cunn" then
    require "cunn"
  end
elseif args.b == "cudnn" then
  require "cunn"
  require "cudnn"
  assert(cudnn.version >= 4000, "cuDNN v4 or higher is required")
  SpatialConvolution = cudnn.SpatialConvolution
  SpatialMaxPooling = cudnn.SpatialMaxPooling
  ReLU = cudnn.ReLU
  SoftMax = cudnn.SoftMax
  SpatialBatchNormalization = cudnn.SpatialBatchNormalization
else
  error("Unknown backend "..args.b)
end

-- Adds to `net` a convolution - Batch Normalization - ReLU series
-- gname is the TensorFlow Graph Google Name of the series
local function ConvBN(gname, net)
  local h5f = hdf5.open(pl.path.join(args.i, gname..".h5"), 'r')
  local strides = h5f:read("strides"):all()
  local padding = h5f:read("padding"):all()
  -- TensorFlow weight matrix is of order: height x width x input_channels x output_channels
  -- make it Torch-friendly: output_channels x input_channels x height x width
  local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
  local ich, och = weights:size(2), weights:size(1)
  local kH, kW = weights:size(3), weights:size(4)

  print(string.format("%s: %d -> %d, kernel (%dx%d), strides (%d, %d), padding (%d, %d)",
    gname, ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1]))

  local conv = SpatialConvolution(ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1])
  conv.weight:copy(weights)
  -- IMPORTANT: there are no biases in the convolutions
  if args.b == "cudnn" then
    conv:noBias()
  else
    conv.bias:zero()
  end
  net:add(conv)

  local bn = SpatialBatchNormalization(och, std_epsilon, nil, true)
  local beta = h5f:read("beta"):all()
  local gamma = h5f:read("gamma"):all()
  local mean = h5f:read("mean"):all()
  local std = h5f:read("std"):all()
  bn.running_mean:copy(mean)
  if args.b == "cudnn" then
    bn.running_std:copy(std:add(std_epsilon):sqrt():pow(-1))
  else
    bn.running_var:copy(std)
  end
  bn.weight:copy(gamma)
  bn.bias:copy(beta)
  net:add(bn)

  net:add(ReLU(true))
  h5f:close()
end

-- Adds to `net` Spatial Pooling, either Max or Average
local function Pool(gname, net)
  local h5f = hdf5.open(pl.path.join(args.i, gname..".h5"), 'r')
  local strides = h5f:read("strides"):all()
  local padding = h5f:read("padding"):all()
  local ksize = h5f:read("ksize"):all()
  local ismax = h5f:read("ismax"):all()
  if ismax[1]==1 then
    print(string.format("%s(Max): (%dx%d), strides (%d, %d), padding (%d, %d)",
      gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
    net:add( SpatialMaxPooling(ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]) )
  else
    print(string.format("%s(Avg): (%dx%d), strides (%d, %d), padding (%d, %d)",
      gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
    net:add(nn.SpatialAveragePooling(
      ksize[3], ksize[2],
      strides[3], strides[2],
      padding[2], padding[1]):setCountExcludePad())
  end
end

-- Adds to `net` Final SoftMax (and its weights) layer
local function Softmax(net)
  local h5f = hdf5.open(pl.path.join(args.i, "softmax.h5"), 'r')
  local weights = h5f:read("weights"):all():permute(2, 1)
  local biases = h5f:read("biases"):all()

  net:add(nn.View(-1):setNumInputDims(3))
  local m = nn.Linear(weights:size(2), weights:size(1))
  m.weight:copy(weights)
  m.bias:copy(biases)
  net:add(m)
  net:add(SoftMax())
end

-- Creates an Inception Branch (SubNetwork), usually called Towers
-- trailing_net is optional and adds trailing network at the end of the tower
local function Tower(names, trailing_net)
  local tower = nn.Sequential()
  for i=1,#names do
    -- separate convolutions / poolings
    if string.find(names[i], "pool") then
      Pool(names[i], tower)
    else
      ConvBN(names[i], tower)
    end
  end
  if trailing_net then
    tower:add(trailing_net)
  end
  return tower
end

-- Creates the suitable branching to host towers
local function Inception(net, towers)
  local concat = nn.DepthConcat(2)
  for i=1,#towers do
    concat:add(towers[i])
  end
  net:add(concat)
end


local net = nn.Sequential()

print("Adding first convolutional layers:")
ConvBN("conv", net)
ConvBN("conv_1", net)
ConvBN("conv_2", net)
Pool("pool", net)
ConvBN("conv_3", net)
ConvBN("conv_4", net)
Pool("pool_1", net)

print("\nAdding Inception 1:")
Inception(net,
  {
    Tower({"mixed_conv"}),
    Tower({"mixed_tower_conv", "mixed_tower_conv_1"}),
    Tower({"mixed_tower_1_conv", "mixed_tower_1_conv_1", "mixed_tower_1_conv_2"}),
    Tower({"mixed_tower_2_pool", "mixed_tower_2_conv"})
  }
)

print("\nAdding Inception 2:")
Inception(net,
  {
    Tower({"mixed_1_conv"}),
    Tower({"mixed_1_tower_conv", "mixed_1_tower_conv_1"}),
    Tower({"mixed_1_tower_1_conv", "mixed_1_tower_1_conv_1", "mixed_1_tower_1_conv_2"}),
    Tower({"mixed_1_tower_2_pool", "mixed_1_tower_2_conv"})
  }
)

print("\nAdding Inception 3:")
Inception(net,
  {
    Tower({"mixed_2_conv"}),
    Tower({"mixed_2_tower_conv", "mixed_2_tower_conv_1"}),
    Tower({"mixed_2_tower_1_conv", "mixed_2_tower_1_conv_1", "mixed_2_tower_1_conv_2"}),
    Tower({"mixed_2_tower_2_pool", "mixed_2_tower_2_conv"})
  }
)

print("\nAdding Inception 4:")
Inception(net,
  {
    Tower({"mixed_3_conv"}),
    Tower({"mixed_3_tower_conv", "mixed_3_tower_conv_1", "mixed_3_tower_conv_2"}),
    Tower({"mixed_3_pool"})
  }
)

print("\nAdding Inception 5:")
Inception(net,
  {
    Tower({"mixed_4_conv"}),
    Tower({"mixed_4_tower_conv", "mixed_4_tower_conv_1", "mixed_4_tower_conv_2"}),
    Tower({"mixed_4_tower_1_conv", "mixed_4_tower_1_conv_1", "mixed_4_tower_1_conv_2", "mixed_4_tower_1_conv_3", "mixed_4_tower_1_conv_4"}),
    Tower({"mixed_4_tower_2_pool", "mixed_4_tower_2_conv"})
  }
)

print("\nAdding Inception 6:")
Inception(net,
  {
    Tower({"mixed_5_conv"}),
    Tower({"mixed_5_tower_conv", "mixed_5_tower_conv_1", "mixed_5_tower_conv_2"}),
    Tower({"mixed_5_tower_1_conv", "mixed_5_tower_1_conv_1", "mixed_5_tower_1_conv_2", "mixed_5_tower_1_conv_3", "mixed_5_tower_1_conv_4"}),
    Tower({"mixed_5_tower_2_pool", "mixed_5_tower_2_conv"})
  }
)

print("\nAdding Inception 7:")
Inception(net,
  {
    Tower({"mixed_6_conv"}),
    Tower({"mixed_6_tower_conv", "mixed_6_tower_conv_1", "mixed_6_tower_conv_2"}),
    Tower({"mixed_6_tower_1_conv", "mixed_6_tower_1_conv_1", "mixed_6_tower_1_conv_2", "mixed_6_tower_1_conv_3", "mixed_6_tower_1_conv_4"}),
    Tower({"mixed_6_tower_2_pool", "mixed_6_tower_2_conv"})
  }
)

print("\nAdding Inception 8:")
Inception(net,
  {
    Tower({"mixed_7_conv"}),
    Tower({"mixed_7_tower_conv", "mixed_7_tower_conv_1", "mixed_7_tower_conv_2"}),
    Tower({"mixed_7_tower_1_conv", "mixed_7_tower_1_conv_1", "mixed_7_tower_1_conv_2", "mixed_7_tower_1_conv_3", "mixed_7_tower_1_conv_4"}),
    Tower({"mixed_7_tower_2_pool", "mixed_7_tower_2_conv"})
  }
)

print("\nAdding Inception 9:")
Inception(net,
  {
    Tower({"mixed_8_tower_conv", "mixed_8_tower_conv_1"}),
    Tower({"mixed_8_tower_1_conv", "mixed_8_tower_1_conv_1", "mixed_8_tower_1_conv_2", "mixed_8_tower_1_conv_3"}),
    Tower({"mixed_8_pool"})
  }
)

print("\nAdding Inception 10:")
-- Note that in the last two Inceptions we have "Inception in Inception" cases
local incept1, incept2 = nn.Sequential(), nn.Sequential()
Inception(incept1,
  {
    Tower({"mixed_9_tower_mixed_conv"}),
    Tower({"mixed_9_tower_mixed_conv_1"})
  }
)
Inception(incept2,
  {
    Tower({"mixed_9_tower_1_mixed_conv"}),
    Tower({"mixed_9_tower_1_mixed_conv_1"})
  }
)
Inception(net,
  {
    Tower({"mixed_9_conv"}),
    Tower({"mixed_9_tower_conv"}, incept1),
    Tower({"mixed_9_tower_1_conv", "mixed_9_tower_1_conv_1"}, incept2),
    Tower({"mixed_9_tower_2_pool", "mixed_9_tower_2_conv"})
  }
)

print("\nAdding Inception 11:")
incept1, incept2 = nn.Sequential(), nn.Sequential()
Inception(incept1,
  {
    Tower({"mixed_10_tower_mixed_conv"}),
    Tower({"mixed_10_tower_mixed_conv_1"})
  }
)
Inception(incept2,
  {
    Tower({"mixed_10_tower_1_mixed_conv"}),
    Tower({"mixed_10_tower_1_mixed_conv_1"})
  }
)
Inception(net,
  {
    Tower({"mixed_10_conv"}),
    Tower({"mixed_10_tower_conv"}, incept1),
    Tower({"mixed_10_tower_1_conv", "mixed_10_tower_1_conv_1"}, incept2),
    Tower({"mixed_10_tower_2_pool", "mixed_10_tower_2_conv"})
  }
)

print("\nAdding Pooling and SoftMax:")
Pool("pool_3", net)
Softmax(net)

if args.b == "cunn" or args.b == 'cudnn' then
  net = net:cuda()
end
net:evaluate()
torch.save(args.o, net, "binary")

print("Done, network saved in ".. args.o)
