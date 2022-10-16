import configargparse

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general config
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--model_type', type=str, choices=['GAN', 'VAE', 'NADE'], help='model type')
    parser.add_argument('--dataset_type', type=str, choices=['MNIST'], help='dataset type')
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--log_path', type=str, default="logs", help='log path')
    # model config
    parser.add_argument('--z_size', type=int, help='GAN and VAE latent vector size')
    parser.add_argument('--h_size', type=int, help='NADE hidden layer size')
    # train config
    parser.add_argument('--gen_learning_rate', type=float, help='GAN generator learning rate')
    parser.add_argument('--dis_learning_rate', type=float, help='GAN discriminator learning rate')
    parser.add_argument('--enc_learning_rate', type=float, help='VAE encoder learning rate')
    parser.add_argument('--dec_learning_rate', type=float, help='VAE decoder learning rate')
    parser.add_argument('--learning_rate', type=float, help='NADE learning rate')
    parser.add_argument('--gen_gamma', type=float, help='GAN generator learning rate decay')
    parser.add_argument('--dis_gamma', type=float, help='GAN discriminator learning rate decay')
    parser.add_argument('--enc_gamma', type=float, help='VAE encoder learning rate decay')
    parser.add_argument('--dec_gamma', type=float, help='VAE decoder learning rate decay')
    parser.add_argument('--gamma', type=float, help='NADE learning rate decay')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--k', type=int, help='number of discriminator training steps per generator training step in GAN')
    parser.add_argument('--eval_num', type=int, help='number of evaluation images')
    parser.add_argument('--eval_interval', type=int, help='evaluation interval')
    
    args = parser.parse_args()
    return args
