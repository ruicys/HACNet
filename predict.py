import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import write_img, chw_to_hwc
from datasets.loader import SingleLoader
from models import *



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='HAFNet-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

parser.add_argument('--datasets_dir', default='./data', type=str, help='path to datasets dir')
parser.add_argument('--train_dataset', default='ITS', type=str, help='train dataset name')
parser.add_argument('--valid_dataset', default='SOTS', type=str, help='valid dataset name')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu




def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(result_dir, exist_ok=True)

	for batch in tqdm(test_loader):
		input = batch['img'].cuda()
		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input).clamp_(-1, 1)
			output = output * 0.5 + 0.5		# [-1, 1] to [0, 1]

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		# print(os.path.join(result_dir, filename))
		write_img(os.path.join(result_dir, filename), out_img)
		

if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	print(saved_model_dir)
	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.folder)

	print(dataset_dir)
	test_dataset = SingleLoader(dataset_dir)
	test_loader = DataLoader(test_dataset,batch_size=1,num_workers=args.num_workers,pin_memory=True)
	result_dir = os.path.join(args.result_dir, args.folder, args.model)
	test(test_loader, network, result_dir)
