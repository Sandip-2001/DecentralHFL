import os
import sys
sys.path.append("..")
from Dataset.utils import init_logs, get_augmix_private_dataloader, get_augmixcorrupt_randompub_dataloader, init_nets, generate_public_data_indexs, mkdirs
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# ----------------------------------------------------------------------------------------------------------------
# --- WEB3 BLOCKCHAIN SETUP ---
from ic.client import Client
from ic.identity import Identity
from ic.agent import Agent
from ic.candid import encode, Types

CANISTER_ID = "uxrrr-q7777-77774-qaaaq-cai"
ic_client = Client(url="http://127.0.0.1:4943")
ic_agent = Agent(identity=Identity(), client=ic_client)

def format_candid_args(arr, p_aug1_arr, p_aug2_arr):
    """Sends logits + both augmented distributions. KL computed on-chain."""
    return encode([
        {'type': Types.Vec(Types.Float64), 'value': arr},
        {'type': Types.Vec(Types.Float64), 'value': p_aug1_arr},
        {'type': Types.Vec(Types.Float64), 'value': p_aug2_arr},
    ])



def decentralized_aggregate(target_list, p_aug1_list, p_aug2_list):
    original_shape = target_list[0].shape

    # Flatten AFTER clamping
    list1 = target_list[0].flatten().tolist()
    list2 = target_list[1].flatten().tolist()
    list3 = target_list[2].flatten().tolist()
    list4 = target_list[3].flatten().tolist()

    # NEW — pass p_aug1 and p_aug2 instead of kl_score
# Flatten the aug distributions to lists
    p_aug1_1 = p_aug1_list[0].flatten().tolist()
    p_aug1_2 = p_aug1_list[1].flatten().tolist()
    p_aug1_3 = p_aug1_list[2].flatten().tolist()
    p_aug1_4 = p_aug1_list[3].flatten().tolist()

    p_aug2_1 = p_aug2_list[0].flatten().tolist()
    p_aug2_2 = p_aug2_list[1].flatten().tolist()
    p_aug2_3 = p_aug2_list[2].flatten().tolist()
    p_aug2_4 = p_aug2_list[3].flatten().tolist()

    ic_agent.update_raw(CANISTER_ID, "client1", format_candid_args(list1, p_aug1_1, p_aug2_1))
    ic_agent.update_raw(CANISTER_ID, "client2", format_candid_args(list2, p_aug1_2, p_aug2_2))
    ic_agent.update_raw(CANISTER_ID, "client3", format_candid_args(list3, p_aug1_3, p_aug2_3))
    ic_agent.update_raw(CANISTER_ID, "client4", format_candid_args(list4, p_aug1_4, p_aug2_4))

    result = ic_agent.query_raw(CANISTER_ID, "testAverage", encode([]))
    averaged_list = result[0]['value']

    return torch.tensor(averaged_list, dtype=torch.float32).view(original_shape).to(device)

# ----------------------------------------------------------------------------------------------------------------
# ================================================================
# ATTACK CONFIGURATION — Change these flags to switch experiments
# ================================================================
# Experiment 1: ENABLE_ATTACK=False, USE_KL_ATTACK=False  → Clean Baseline
# Experiment 2: ENABLE_ATTACK=True,  USE_KL_ATTACK=False  → Byzantine Logit Attack
# Experiment 3: ENABLE_ATTACK=True,  USE_KL_ATTACK=True   → Byzantine + KL Score Inflation

USE_BLOCKCHAIN    = True        # True = Internet Computer | False = Central server
ENABLE_ATTACK    = True       # Set False for clean baseline run
USE_KL_ATTACK    = False      # Set True to also inflate KL scores (combined attack)
ATTACKER_INDICES = [0, 1]     # Which client indices are malicious (0,1 = 50% Byzantine)
ATTACK_TYPE      = "sign_flip"  # Options: "sign_flip", "gaussian_noise", "zero_gradient"
KL_INFLATE_FACTOR = 1000.0    # How much to inflate the malicious clients' KL scores


def inject_byzantine_attack(target_list, attacker_indices, attack_type="gaussian_noise", scale=10.0):
    """
    Simulates Byzantine clients sending poisoned logits to the aggregation server.
    - 'sign_flip':      Flips + scales predictions → pulls consensus in wrong direction
    - 'gaussian_noise': Replaces predictions with random noise → disrupts consensus
    - 'zero_gradient':  Sends all zeros → client effectively disappears
    """
    poisoned = [t.clone() for t in target_list]
    for idx in attacker_indices:
        if attack_type == "sign_flip":
            poisoned[idx] = -target_list[idx] * scale
        elif attack_type == "gaussian_noise":
            poisoned[idx] = torch.randn_like(target_list[idx]) * scale
        elif attack_type == "zero_gradient":
            poisoned[idx] = torch.zeros_like(target_list[idx])
    return poisoned


def inject_kl_manipulation_attack(kl_list, attacker_indices, inflate_factor=100.0):
    """
    Malicious clients claim an artificially inflated trust (KL) score
    while sending poisoned logits — making the blockchain over-weight their bad updates.
    This specifically targets the weighted consensus in testAverage().
    """
    poisoned_kl = [k.clone() if torch.is_tensor(k) else k for k in kl_list]
    for idx in attacker_indices:
        poisoned_kl[idx] = poisoned_kl[idx] * inflate_factor
    return poisoned_kl
# ================================================================


# --- APPLE SILICON SUPPORT ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"Attack config → ENABLE_ATTACK={ENABLE_ATTACK}, USE_KL_ATTACK={USE_KL_ATTACK}, "
      f"TYPE={ATTACK_TYPE}, ATTACKERS={ATTACKER_INDICES}")

# ----------------------------------------------------------------------------------------------------------------
'''
Global Parameters
'''
Seed = 0
N_Participants = 4
TrainBatchSize = 128           
TestBatchSize = 256           
CommunicationEpoch = 20       # Increased from 2 so attack effects accumulate visibly
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}

"""Corruption Setting"""
Corruption_Type  = 'random_noise'
Corrupt_rate     = 1
Test_Corrupt_rate = 1
if Test_Corrupt_rate == 0:
    test_dataset = 'clean'
else:
    test_dataset = 'corrupt'

"""Heterogeneous Model Setting"""
Private_Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']

"""Dataset Setting"""
if Corrupt_rate == 0:
    Private_Dataset_Name = 'cifar10'
    Private_Dataset_Dir  = '../Dataset/cifar_10'
else:
    Private_Dataset_Name = 'cifar10c'
    Private_Dataset_Dir  = '../Dataset/cifar_10/CIFAR-10-C_train'

Private_Data_Len = 2000   
Private_Dataset_Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Private_Output_Channel  = len(Private_Dataset_Classes)

"""Public Dataset Setting"""
Public_Dataset_Name   = 'cifar100'
Public_Dataset_Dir    = '../Dataset/cifar_100/CIFAR-100-C_train'
Public_Dataset_Length = 1000  


def evaluate_network(network, dataloader, logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total   = 0
        for images, labels in dataloader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs, _ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy: {} %'.format(acc))
    return acc


def update_model_via_private_data(network, private_epoch, private_dataloader,
                                  loss_function, optimizer_method, learing_rate, logger):
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    if optimizer_method == 'Adam':
        optimizer = optim.Adam(network.parameters(), lr=learing_rate)
    if optimizer_method == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr=learing_rate, momentum=0.9, weight_decay=1e-4)

    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images_all = torch.cat(images, 0).to(device)
            labels     = labels.to(device)
            logits_all, _ = network(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
            loss = criterion(logits_clean, labels.long())
            p_clean, p_aug1, p_aug2 = (F.softmax(logits_clean, dim=1),
                                       F.softmax(logits_aug1,  dim=1),
                                       F.softmax(logits_aug2,  dim=1))
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1,  reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2,  reduction='batchmean')) / 3.
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 == 0:
                logger.info('Private Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(images), len(private_dataloader.dataset),
                    100. * batch_idx / len(private_dataloader), loss.item()))
    return network, participant_local_loss_batch_list


if __name__ == '__main__':
    logger = init_logs()
    logger_global = logger
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NOTE: DataParallel removed — MPS does not support multi-GPU DataParallel
    # All network.to(device) calls use single-device MPS

    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs and Model")
    net_dataidx_map = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(50000)
        idxes = idxes[0:Private_Data_Len]
        net_dataidx_map[index] = idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=N_Participants, nets_name_list=Private_Nets_Name_List,
                         num_classes=Private_Output_Channel)
    logger.info("Load Participants' Models")

    for i in range(N_Participants):
        network = net_list[i]
        network = network.to(device)   # ← DataParallel removed
        netname = Private_Nets_Name_List[i]
        # Load the checkpoint
        ckpt_path = ('../Network/Model_Storage/' + Corruption_Type + '_' + str(Corrupt_rate) +
                 '_' + Pariticpant_Params['loss_funnction'] + '/' + netname + '_' + str(i) + '.ckpt')
        state_dict = torch.load(ckpt_path, map_location=device)

        # Strip the 'module.' prefix added by DataParallel when the author saved the models
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '', 1)  # remove only the first 'module.' prefix
            new_state_dict[new_key] = v

        network.load_state_dict(new_state_dict)
        net_list[i] = network

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(
        dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir, size=Public_Dataset_Length)
    public_train_dl, _, public_train_ds, _ = get_augmixcorrupt_randompub_dataloader(
        dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
        train_bs=TrainBatchSize, test_bs=TestBatchSize, dataidxs=public_data_indexs,
        test_dataset=test_dataset, test_corrupt_rate=Test_Corrupt_rate)

    col_loss_list   = []
    local_loss_list = []
    acc_list        = []

    for epoch_index in range(CommunicationEpoch):
        logger.info("=== Communication Epoch " + str(epoch_index) + " ===")

        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname             = Private_Nets_Name_List[participant_index]
            private_dataset_dir = Private_Dataset_Dir
            _, test_dl, _, _ = get_augmix_private_dataloader(
                dataset=Private_Dataset_Name, datadir=private_dataset_dir,
                train_bs=TrainBatchSize, test_bs=TestBatchSize, dataidxs=None,
                corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                test_dataset=test_dataset, test_corrupt_rate=Test_Corrupt_rate)
            network  = net_list[participant_index]
            network  = network.to(device)   # ← DataParallel removed
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants
        logger.info(f"[Epoch {epoch_index}] Mean Accuracy: {accuracy_avg:.2f}% | "
                    f"Attack: {ATTACK_TYPE if ENABLE_ATTACK else 'None'} | "
                    f"KL Attack: {USE_KL_ATTACK} | Attackers: {ATTACKER_INDICES if ENABLE_ATTACK else []}")

        '''
        HHF — Public Data Aggregation via Blockchain
        '''
        for batch_idx, (images, _) in enumerate(public_train_dl):
            p_aug1_collect = []
            p_aug2_collect = []
            linear_output_list        = []
            linear_output_target_list = []
            kl_loss_batch_list        = []
            participant_kl_list       = []

            # --- Step 1: Calculate clean logits and KL scores from all clients ---
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = network.to(device)   # ← DataParallel removed
                network.train()
                image      = images[participant_index]
                images_all = torch.cat(image, 0).to(device)
                logits_all, _ = network(x=images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, image[0].size(0))
                p_clean, p_aug1, p_aug2 = (F.softmax(logits_clean, dim=1),
                                           F.softmax(logits_aug1,  dim=1),
                                           F.softmax(logits_aug2,  dim=1))
                p_aug1_collect.append(p_aug1.detach())
                p_aug2_collect.append(p_aug2.detach())
                plog_clean = p_clean.log()
                linear_output_target_list.append(p_clean.detach())
                linear_output_list.append(plog_clean)
                participant_kl = 1 / (
                    F.kl_div(p_aug1.log(), p_clean, reduction='batchmean') +
                    F.kl_div(p_aug2.log(), p_clean, reduction='batchmean'))
                participant_kl_list.append(participant_kl.cpu().clone().detach())

            # ================================================================
            # --- Step 2: INJECT ATTACKS (if enabled) ---
            # This block sits between logit collection and blockchain aggregation.
            # Simulates what a malicious client would send to the server.

            if ENABLE_ATTACK:
                # Apply Byzantine logit poisoning
                aggregation_targets = inject_byzantine_attack(
                    linear_output_target_list, ATTACKER_INDICES, ATTACK_TYPE
                )
                if USE_KL_ATTACK:
                    # Additionally inflate the attackers' trust scores
                    # → tests whether your blockchain's weighted consensus is exploitable
                    aggregation_kl = inject_kl_manipulation_attack(
                        participant_kl_list, ATTACKER_INDICES, KL_INFLATE_FACTOR
                    )
                else:
                    aggregation_kl = participant_kl_list
            else:
                # Clean run — no attack
                aggregation_targets = linear_output_target_list
                aggregation_kl      = participant_kl_list
            # ================================================================

            # STEP 3 — Aggregate: blockchain OR central server
            if USE_BLOCKCHAIN:
                # ── Internet Computer blockchain aggregation ──────────────────────
                global_consensus_logits = decentralized_aggregate(aggregation_targets, p_aug1_collect, p_aug2_collect)
            else:
                # ── Central server aggregation (original AugHFL weighted average) ─
                kl_sum = sum(aggregation_kl) + 1e-7
                size   = aggregation_targets[0].shape[0]
                global_consensus_logits = torch.zeros_like(aggregation_targets[0])
                for i in range(N_Participants):
                    weight = aggregation_kl[i] / kl_sum
                    global_consensus_logits += aggregation_targets[i] * weight

            # --- Step 4: Each client updates locally against the blockchain consensus ---
            for participant_index in range(N_Participants):
                network   = net_list[participant_index]
                network   = network.to(device)   # ← DataParallel removed
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()
                # my_logits = torch.clamp(linear_output_list[participant_index], min=1e-7, max=1.0)
                my_logits = linear_output_list[participant_index]
                loss      = criterion(my_logits, global_consensus_logits)
                kl_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()
            col_loss_list.append(kl_loss_batch_list)

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = network.to(device)   # ← DataParallel removed
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_augmix_private_dataloader(
                dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                train_bs=TrainBatchSize, test_bs=TestBatchSize,
                dataidxs=private_dataidx, corrupt_type=Corruption_Type,
                corrupt_rate=Corrupt_rate, test_dataset=test_dataset,
                test_corrupt_rate=Test_Corrupt_rate)
            private_epoch = max([int(len(public_train_ds) / len(train_ds_local)), 1])
            network, private_loss_batch_list = update_model_via_private_data(
                network=network, private_epoch=private_epoch,
                private_dataloader=train_dl_local,
                loss_function=Pariticpant_Params['loss_funnction'],
                optimizer_method=Pariticpant_Params['optimizer_name'],
                learing_rate=Pariticpant_Params['learning_rate'],
                logger=logger)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        """Final round evaluation"""
        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):
                _, test_dl, _, _ = get_augmix_private_dataloader(
                    dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                    train_bs=TrainBatchSize, test_bs=TestBatchSize, dataidxs=None,
                    corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                    test_dataset=test_dataset, test_corrupt_rate=Test_Corrupt_rate)
                network  = net_list[participant_index]
                network  = network.to(device)   # ← DataParallel removed
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / N_Participants

        if epoch_index % 5 == 0 or epoch_index == CommunicationEpoch - 1:
            # Include attack config in saved model path so experiments don't overwrite each other
            attack_tag = f"attack_{ATTACK_TYPE}" if ENABLE_ATTACK else "clean"
            save_dir   = (f'./test/Model_Storage/{Corruption_Type}_{Corrupt_rate}_'
                          f'{Pariticpant_Params["loss_funnction"]}_{attack_tag}')
            mkdirs(save_dir)
            logger.info('Save Models to ' + save_dir)
            for participant_index in range(N_Participants):
                netname = Private_Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = network.to(device)   # ← DataParallel removed
                torch.save(network.state_dict(),
                           save_dir + '/' + netname + '_' + str(participant_index) + '.ckpt')
