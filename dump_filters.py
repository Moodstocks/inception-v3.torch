import os.path
import re
import sys
import tarfile

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile
import h5py
import math

paddings = {"VALID": [0, 0], "SAME": [1, 1]}

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long



def create_graph():
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def make_padding(padding_name, conv_shape):
  if padding_name == "VALID":
    return [0, 0]
  elif padding_name == "SAME":
    return [int(math.ceil(conv_shape[0]/2)), int(math.ceil(conv_shape[1]/2))]
  else:
    sys.exit('Invalid padding name '+padding_name)


def dump_convbn(sess, gname):
  conv = sess.graph.get_operation_by_name(gname + '/Conv2D')

  weights = sess.graph.get_tensor_by_name(gname + '/conv2d_params:0').eval()
  padding = make_padding(conv.get_attr("padding"), weights.shape)
  strides = conv.get_attr("strides")

  beta = sess.graph.get_tensor_by_name(gname + '/batchnorm/beta:0').eval()
  gamma = sess.graph.get_tensor_by_name(gname + '/batchnorm/gamma:0').eval()
  mean = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_mean:0').eval()
  std = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_variance:0').eval()

  gname = gname.replace("/", "_")
  h5f = h5py.File('dump/'+gname+'.h5', 'w')
  h5f.create_dataset("weights", data=weights)
  h5f.create_dataset("strides", data=strides)
  h5f.create_dataset("padding", data=padding)
  h5f.create_dataset("beta", data=beta)
  h5f.create_dataset("gamma", data=gamma)
  h5f.create_dataset("mean", data=mean)
  h5f.create_dataset("std", data=std)
  h5f.close()

def dump_pool(sess, gname):
  pool = sess.graph.get_operation_by_name(gname)
  ismax = pool.type=='MaxPool' and 1 or 0
  ksize = pool.get_attr("ksize")
  padding = make_padding(pool.get_attr("padding"), ksize[1:3])
  strides = pool.get_attr("strides")

  gname = gname.replace("/", "_")
  h5f = h5py.File('dump/'+gname+'.h5', 'w')
  h5f.create_dataset("ismax", data=[ismax])
  h5f.create_dataset("ksize", data=ksize)
  h5f.create_dataset("padding", data=padding)
  h5f.create_dataset("strides", data=strides)
  h5f.close()

def dump_softmax(sess):
  softmax_w = sess.graph.get_tensor_by_name('softmax/weights:0').eval()
  softmax_b = sess.graph.get_tensor_by_name('softmax/biases:0').eval()
  h5f = h5py.File('dump/softmax.h5', 'w')
  h5f.create_dataset("weights", data=softmax_w)
  h5f.create_dataset("biases", data=softmax_b)
  h5f.close()



def run_inference_on_image(image):
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image).read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:

    # Run the graph until softmax
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    # print predictions indices and values
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for p in top_k:
      print(p, predictions[p])

    if not os.path.exists("dump"):
      os.makedirs("dump")

    # dump the filters
    dump_convbn(sess, 'conv')
    dump_convbn(sess, 'conv_1')
    dump_convbn(sess, 'conv_2')
    dump_pool(sess,   'pool')
    dump_convbn(sess, 'conv_3')
    dump_convbn(sess, 'conv_4')
    dump_pool(sess,   'pool_1')

    # inceptions with 1x1, 3x3, 5x5 convolutions
    dump_convbn(sess, 'mixed/conv')
    dump_convbn(sess, 'mixed/tower/conv')
    dump_convbn(sess, 'mixed/tower/conv_1')
    dump_convbn(sess, 'mixed/tower_1/conv')
    dump_convbn(sess, 'mixed/tower_1/conv_1')
    dump_convbn(sess, 'mixed/tower_1/conv_2')
    dump_pool(sess,   'mixed/tower_2/pool')
    dump_convbn(sess, 'mixed/tower_2/conv')

    dump_convbn(sess, 'mixed_1/conv')
    dump_convbn(sess, 'mixed_1/tower/conv')
    dump_convbn(sess, 'mixed_1/tower/conv_1')
    dump_convbn(sess, 'mixed_1/tower_1/conv')
    dump_convbn(sess, 'mixed_1/tower_1/conv_1')
    dump_convbn(sess, 'mixed_1/tower_1/conv_2')
    dump_pool(sess,   'mixed_1/tower_2/pool')
    dump_convbn(sess, 'mixed_1/tower_2/conv')

    dump_convbn(sess, 'mixed_2/conv')
    dump_convbn(sess, 'mixed_2/tower/conv')
    dump_convbn(sess, 'mixed_2/tower/conv_1')
    dump_convbn(sess, 'mixed_2/tower_1/conv')
    dump_convbn(sess, 'mixed_2/tower_1/conv_1')
    dump_convbn(sess, 'mixed_2/tower_1/conv_2')
    dump_pool(sess,   'mixed_2/tower_2/pool')
    dump_convbn(sess, 'mixed_2/tower_2/conv')

    # inceptions with 1x1, 3x3(in sequence) convolutions
    dump_convbn(sess, 'mixed_3/conv')
    dump_convbn(sess, 'mixed_3/tower/conv')
    dump_convbn(sess, 'mixed_3/tower/conv_1')
    dump_convbn(sess, 'mixed_3/tower/conv_2')
    dump_pool(sess,   'mixed_3/pool')

    # inceptions with 1x1, 7x1, 1x7 convolutions
    dump_convbn(sess, 'mixed_4/conv')
    dump_convbn(sess, 'mixed_4/tower/conv')
    dump_convbn(sess, 'mixed_4/tower/conv_1')
    dump_convbn(sess, 'mixed_4/tower/conv_2')
    dump_convbn(sess, 'mixed_4/tower_1/conv')
    dump_convbn(sess, 'mixed_4/tower_1/conv_1')
    dump_convbn(sess, 'mixed_4/tower_1/conv_2')
    dump_convbn(sess, 'mixed_4/tower_1/conv_3')
    dump_convbn(sess, 'mixed_4/tower_1/conv_4')
    dump_pool(sess,   'mixed_4/tower_2/pool')
    dump_convbn(sess, 'mixed_4/tower_2/conv')

    dump_convbn(sess, 'mixed_5/conv')
    dump_convbn(sess, 'mixed_5/tower/conv')
    dump_convbn(sess, 'mixed_5/tower/conv_1')
    dump_convbn(sess, 'mixed_5/tower/conv_2')
    dump_convbn(sess, 'mixed_5/tower_1/conv')
    dump_convbn(sess, 'mixed_5/tower_1/conv_1')
    dump_convbn(sess, 'mixed_5/tower_1/conv_2')
    dump_convbn(sess, 'mixed_5/tower_1/conv_3')
    dump_convbn(sess, 'mixed_5/tower_1/conv_4')
    dump_pool(sess,   'mixed_5/tower_2/pool')
    dump_convbn(sess, 'mixed_5/tower_2/conv')

    dump_convbn(sess, 'mixed_6/conv')
    dump_convbn(sess, 'mixed_6/tower/conv')
    dump_convbn(sess, 'mixed_6/tower/conv_1')
    dump_convbn(sess, 'mixed_6/tower/conv_2')
    dump_convbn(sess, 'mixed_6/tower_1/conv')
    dump_convbn(sess, 'mixed_6/tower_1/conv_1')
    dump_convbn(sess, 'mixed_6/tower_1/conv_2')
    dump_convbn(sess, 'mixed_6/tower_1/conv_3')
    dump_convbn(sess, 'mixed_6/tower_1/conv_4')
    dump_pool(sess,   'mixed_6/tower_2/pool')
    dump_convbn(sess, 'mixed_6/tower_2/conv')

    dump_convbn(sess, 'mixed_7/conv')
    dump_convbn(sess, 'mixed_7/tower/conv')
    dump_convbn(sess, 'mixed_7/tower/conv_1')
    dump_convbn(sess, 'mixed_7/tower/conv_2')
    dump_convbn(sess, 'mixed_7/tower_1/conv')
    dump_convbn(sess, 'mixed_7/tower_1/conv_1')
    dump_convbn(sess, 'mixed_7/tower_1/conv_2')
    dump_convbn(sess, 'mixed_7/tower_1/conv_3')
    dump_convbn(sess, 'mixed_7/tower_1/conv_4')
    dump_pool(sess,   'mixed_7/tower_2/pool')
    dump_convbn(sess, 'mixed_7/tower_2/conv')

    # inceptions with 1x1, 3x3, 1x7, 7x1 filters
    dump_convbn(sess, 'mixed_8/tower/conv')
    dump_convbn(sess, 'mixed_8/tower/conv_1')
    dump_convbn(sess, 'mixed_8/tower_1/conv')
    dump_convbn(sess, 'mixed_8/tower_1/conv_1')
    dump_convbn(sess, 'mixed_8/tower_1/conv_2')
    dump_convbn(sess, 'mixed_8/tower_1/conv_3')
    dump_pool(sess,   'mixed_8/pool')

    dump_convbn(sess, 'mixed_9/conv')
    dump_convbn(sess, 'mixed_9/tower/conv')
    dump_convbn(sess, 'mixed_9/tower/mixed/conv')
    dump_convbn(sess, 'mixed_9/tower/mixed/conv_1')
    dump_convbn(sess, 'mixed_9/tower_1/conv')
    dump_convbn(sess, 'mixed_9/tower_1/conv_1')
    dump_convbn(sess, 'mixed_9/tower_1/mixed/conv')
    dump_convbn(sess, 'mixed_9/tower_1/mixed/conv_1')
    dump_pool(sess,   'mixed_9/tower_2/pool')
    dump_convbn(sess, 'mixed_9/tower_2/conv')

    dump_convbn(sess, 'mixed_10/conv')
    dump_convbn(sess, 'mixed_10/tower/conv')
    dump_convbn(sess, 'mixed_10/tower/mixed/conv')
    dump_convbn(sess, 'mixed_10/tower/mixed/conv_1')
    dump_convbn(sess, 'mixed_10/tower_1/conv')
    dump_convbn(sess, 'mixed_10/tower_1/conv_1')
    dump_convbn(sess, 'mixed_10/tower_1/mixed/conv')
    dump_convbn(sess, 'mixed_10/tower_1/mixed/conv_1')
    dump_pool(sess,   'mixed_10/tower_2/pool')
    dump_convbn(sess, 'mixed_10/tower_2/conv')

    dump_pool(sess, "pool_3")
    dump_softmax(sess)


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)

if __name__ == '__main__':
  tf.app.run()
