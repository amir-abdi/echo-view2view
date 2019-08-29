import json
from absl import app
from absl import flags
from keras import backend as K
import os
from data_loader_camus import DataLoaderCamus
from patch_gan import PatchGAN
import matplotlib

matplotlib.use('Agg')

flags.DEFINE_string('dataset_path', None, 'Path of the dataset.')
flags.DEFINE_string('gpu', '0', 'Comma separated list of GPU cores to use for training.')
flags.DEFINE_boolean('test', False, 'Test model and generate outputs on the test set')
flags.DEFINE_string('config', None, 'Config file for training hyper-parameters.')
flags.DEFINE_boolean('use_wandb', False, 'Use wandb for logging')
flags.DEFINE_string('wandb_resume_id', None, 'Resume wandb process with the given id')
flags.DEFINE_string('ckpt_load', None, 'Path to load the model')
flags.DEFINE_string('seg_load', None, 'Path to load the segmentation model')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('config')

FLAGS = flags.FLAGS


def main(argv):
    # Load configs from file
    config = json.load(open(FLAGS.config))

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    from keras.optimizers import tf
    cf = tf.ConfigProto()
    cf.gpu_options.allow_growth = True
    sess = tf.Session(config=cf)
    K.set_session(sess)

    # Set name
    name = 'F{}_B{}_{}_{}_'.format(config['FIRST_LAYERS_FILTERS'], config['BATCH_SIZE'],
                                   config['INPUT_NAME'], config['TARGET_NAME'])
    for l in config['LABELS']:
        name += str(l)
    config['NAME'] += '_' + name

    # Initialize data loader
    augmentation = dict()
    for key, value in config.items():
        if 'AUG_' in key:
            augmentation[key] = value

    data_loader = DataLoaderCamus(
        dataset_path=FLAGS.dataset_path,
        input_name=config['INPUT_NAME'],
        target_name=config['TARGET_NAME'],
        img_res=config['IMAGE_RES'],
        target_rescale=config['TARGET_TRANS'],
        input_rescale=config['INPUT_TRANS'],
        labels=config['LABELS'],
        train_ratio=0.8,  # Ratio of data used for training
        valid_ratio=0.05,  # Ratio of training data used for validation
        augment=augmentation
    )

    if FLAGS.use_wandb:
        import wandb
        resume_wandb = True if FLAGS.wandb_resume_id is not None else False
        wandb.init(config=config, resume=resume_wandb, id=FLAGS.wandb_resume_id, project='EchoView2View')

    # Initialize GAN
    model = PatchGAN(data_loader, config, FLAGS.use_wandb)

    # load trained models if they exist
    if FLAGS.ckpt_load is not None:
        model.load_model(FLAGS.ckpt_load)

    if FLAGS.test:
        model.test(FLAGS.seg_load)
    else:
        model.train()


if __name__ == '__main__':
    app.run(main)
