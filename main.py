import argparse
import os
import torch
import numpy as np
from defense import EP, BNP, CLP
from models import get_model
from utils import TensorsDataset

def main():
    parser = argparse.ArgumentParser(description="Landseer Wrapper for EP/BNP/CLP Defenses")
    
    # Landseer Required Arguments
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing model.pt and data.npy")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the defended model")
    
    # Defense Hyperparameters
    parser.add_argument('--defense-type', type=str, choices=['EP', 'BNP', 'CLP'], default='EP')
    parser.add_argument('-u', type=float, default=3.0, help="Pruning threshold (k for EP/BNP, u for CLP)")
    parser.add_argument('--model-name', type=str, default='resnet20', help="Model architecture name")
    parser.add_argument('--batch-size', type=int, default=500, help="Batch size for pruning statistics")
    
    args = parser.parse_args()

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 1. Load Data
    # Landseer provides 'data.npy' and 'labels.npy' in the input directory
    data_path = os.path.join(args.input_dir, 'data.npy')
    label_path = os.path.join(args.input_dir, 'labels.npy')
    
    imgs = np.load(data_path)
    labels = np.load(label_path)
    
    # Convert to Tensors and Create Loader
    dataset = TensorsDataset(torch.from_numpy(imgs).float(), torch.from_numpy(labels).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    num_classes = len(np.unique(labels))

    # 2. Load the Model
    # Landseer provides the model weights as 'model.pt'
    model_path = os.path.join(args.input_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=args.device)

    # 3. Execute Defense
    print(f"Running {args.defense_type} defense...")
    
    if args.defense_type == 'CLP':
        # CLP modifies the model in-place
        net = get_model(args.model_name, num_classes).to(args.device)
        net.load_state_dict(state_dict)
        CLP(net, args.u)
        final_state = net.state_dict()
        
    elif args.defense_type == 'EP':
        # EP expects the original state_dict and returns a pruned model
        # We pass args so it can access args.device and args.model
        args.model = args.model_name
        pruned_net = EP(state_dict, args.u, loader, args, num_classes)
        final_state = pruned_net.state_dict()
        
    elif args.defense_type == 'BNP':
        # BNP works similarly to EP
        args.model = args.model_name
        pruned_net = BNP(state_dict, args.u, loader, args, num_classes)
        final_state = pruned_net.state_dict()

    # 4. Save for Landseer
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'defended_model.pt')
    torch.save(final_state, output_path)
    print(f"Success! Defended model saved to {output_path}")

if __name__ == '__main__':
    main()