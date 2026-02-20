import argparse
from trainer.trainer_IRM_v2 import IRMTrainerInterface

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=20)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    # MODEL CONTROLLER
    parser.add_argument("--model", default='training/training_controller/2026-02-20 00-56-05_/checkpoints/model_controller-epoch=90-val_loss=11.7121.ckpt')

    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--saveImages", default=True, help="plot qualitative examples in tensorboard")
    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    t = IRMTrainerInterface(config)
    print('start training IRM')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
