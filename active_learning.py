import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from dataset.phc_dataset import PhC2DBandgapQuickLoad
from train_model import set_seed, create_and_train_model, test

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)

def get_uncertainties_in_order(dataset, model, args):
    inputs = dataset[:][0].to(args.device)
    with torch.no_grad():
        _, _, predictions_variance = model(inputs)
    return predictions_variance # Largest std dev -> More uncertainty

def get_square_errors_in_order(dataset, model, args):
    inputs = dataset[:][0].to(args.device)
    targets = dataset[:][1].to(args.device)
    with torch.no_grad():
        predictions, _, _ = model(inputs)
        square_errors = (predictions-targets)**2
    return square_errors

def eval_prioritization_strategy(train_dataset, test_loader, args):
    """
    Returns test accuracies vs num samples from active learning strategy. prioritizer specifies how samples are selected
    """

    test_accuracies = []
    train_indices = np.random.choice(np.arange(0, len(train_dataset)), args.active_batch_size, replace=False) # Initial sample pool

    num_active_iterations = int(np.ceil(len(train_dataset) / args.active_batch_size))

    for active_iteration in range(num_active_iterations): # each active learning iteration
        logger.info(f'Number of train indices = {len(set(train_indices))}')

        # Create subset of training data
        train_dataset_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_loader_subset = torch.utils.data.DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=True)

        model = create_and_train_model(args, train_loader_subset, test_loader, save=False)
        
        test_accuracies.append(test(args, test_loader, model).cpu().item()) # Add test accuracy

        logger.info(f'-'*40)
        logger.info(f'Test accuracies = {test_accuracies}')
        logger.info(f'-'*40)

        # Save model and indices
        iteration_save_folder = os.path.join(args.out, f'active_iterations_{active_iteration}')
        os.makedirs(iteration_save_folder)
        torch.save({'epoch': args.epochs + 1,
                'state_dict': model.state_dict()}, 
                os.path.join(iteration_save_folder, f'epoch_{args.epochs}_checkpoint.pth.tar'))
        np.save(os.path.join(iteration_save_folder, 'train_indices.npy'), train_indices)
        if args.final_budget is not None:
            if len(test_accuracies) >= int(args.final_budget / args.active_batch_size):
                break

        # Update train indices through prioritizer
        if args.prioritizer == 'uncertain':
            std_devs = get_uncertainties_in_order(train_dataset, model, args)
            train_indices = prioritize_uncertain(train_indices, std_devs, args.active_batch_size)
        elif args.prioritizer == 'error':
            square_errors = get_square_errors_in_order(train_dataset, model, args)
            train_indices = prioritize_uncertain(train_indices, square_errors, args.active_batch_size)
        elif args.prioritizer == 'random':
            train_indices = prioritize_random(train_indices, train_dataset, args.active_batch_size)
        else:
            raise ValueError('Incorrect prioritizer')

    return test_accuracies

def prioritize_uncertain(train_indices, std_devs, active_batch_size):
    std_devs = std_devs.cpu().numpy()
    most_uncertain_samples = np.argsort(-1* std_devs) # indices of largest to smallest std_devs
    most_uncertain_new_samples = [index for index in most_uncertain_samples if index not in train_indices]
    new_train_indices = [*train_indices, *most_uncertain_new_samples[:active_batch_size]]
    return new_train_indices

def prioritize_random(train_indices, train_dataset, active_batch_size):
    all_indices = list(range(len(train_dataset)))
    new_samples = np.array([index for index in all_indices if index not in train_indices])
    num_new_train_indices = min(active_batch_size, len(new_samples))
    new_train_indices = [*train_indices, *np.random.choice(new_samples, num_new_train_indices, replace=False)]
    return new_train_indices

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--lambda-kl', default=1.0, type=float,
                        help="Coefficient for KL loss")
    parser.add_argument('--active-batch-size', default=100, type=int,
                        help="Number of samples added at each active iteration")
    parser.add_argument('--prioritizer', type=str, required=True, choices=['uncertain', 'random', 'error'],
                        help="How to actively choose samples")
    parser.add_argument('--final-budget', type=int, default=None,
                        help='Approximate final budget to stop at (modulo active batch size)')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Skips saving and testing mid training to speed up.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    logging.info(f'Device = {args.device}')
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    train_dataset = PhC2DBandgapQuickLoad(train=True)
    test_dataset = PhC2DBandgapQuickLoad(train=False)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    test_accuracies = eval_prioritization_strategy(train_dataset, test_loader, args)
    logger.info(f'Test accuracies = {test_accuracies}')

    np.save(os.path.join(args.out, 'accuracy.npy'), np.array(test_accuracies))

if __name__ == '__main__':
    main()