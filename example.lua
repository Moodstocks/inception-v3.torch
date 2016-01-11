require 'torch'
require 'image'
require 'nn'
local pl = require('pl.import_into')()

-- Some definitions copied from the TensorFlow model
-- input subtraction
local input_sub = 128
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 299

local load_image = function(path)
  local img   = image.load(path, 3)
  local w, h  = img:size(3), img:size(2)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, input_dim)
  -- normalize image
  img:mul(255):add(-input_sub):mul(input_scale)
  -- due to batch normalization we must use minibatches
  return img:float():view(1, img:size(1), img:size(2), img:size(3))
end

local args = pl.lapp [[
  -m (string) inception v3 model file
  -b (string) backend of the model: "nn"|"cunn"|"cudnn"
  -i (string) image file
  -s (string) synsets file
]]

if args.b == "cunn" then
  require "cunn"
elseif args.b == "cudnn" then
  require "cunn"
  require "cudnn"
end

local net = torch.load(args.m)
net:evaluate()

local synsets = pl.utils.readlines(args.s)

local img = load_image(args.i)
if args.b == "cudnn" or args.b == "cunn" then
  img = img:cuda()
end

-- predict
local scores = net:forward(img)
scores = scores:float():squeeze()

-- find top5 matches
local _,ind = torch.sort(scores, true)
print('\nRESULTS (top-5):')
print('----------------')
for i=1,5 do
  local synidx = ind[i]
  print(string.format(
    "score = %f: %s (%d)", scores[ind[i]], synsets[synidx], ind[i]
  ))
end
