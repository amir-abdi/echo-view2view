import json
from absl import app
from absl import flags
from keras import backend as K

from data_loader_camus import DataLoaderCamus
from patch_gan import PatchGAN

flags.DEFINE_string('dataset_path', None, 'Path of the dataset.')
flags.DEFINE_boolean('test', False, 'Test model and generate outputs on the test set')
flags.DEFINE_string('config', None, 'Config file for training hyper-parameters.')
flags.DEFINE_boolean('use_wandb', False, 'Use wandb for logging')
flags.DEFINE_string('wandb_resume_id', None, 'Resume wandb process with the given id')
flags.DEFINE_string('ckpt_load', None, 'Path to load the model')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('config')

FLAGS = flags.FLAGS


def set_keras_backend(backend):
    print('Available GPUS:', K.tensorflow_backend._get_available_gpus())
    print('Setting backend to {}...'.format(backend))
    if backend == 'tensorflow':
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        K.clear_session()


def main(argv):
    set_keras_backend('tensorflow')

    # Load configs from file
    config = json.load(open(FLAGS.config))

    # Set name
    name = '{}_{}_'.format(config['INPUT_NAME'], config['TARGET_NAME'])
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
        train_ratio=0.05,
        augment=augmentation
    )

    if FLAGS.use_wandb:
        import wandb
        resume_wandb = True if FLAGS.wandb_resume_id is not None else False
        wandb.init(config=config, resume=resume_wandb, id=FLAGS.wandb_resume_id, project='EchoGen')

    # Initialize GAN
    model = PatchGAN(data_loader, config, FLAGS.use_wandb)

    # load trained models if they exist
    if FLAGS.ckpt_load is not None:
        model.load_model(FLAGS.ckpt_load)

    if FLAGS.test:
        model.test()
    else:
        model.train()


if __name__ == '__main__':
    app.run(main)
