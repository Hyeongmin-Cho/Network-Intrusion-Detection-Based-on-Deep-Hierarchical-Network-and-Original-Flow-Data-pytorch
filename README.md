# Network-Intrusion-Detection-Based-on-Deep-Hierarchical-Network-and-Original-Flow-Data-pytorch
A Pytorch Implementation of paper "Network Intrusion Detection: Based on Deep Hierarchical Network and Original Flow Data"


# Dataset
This repository does not offer training dataset.
You have to download and pre-process network intrusion datasets which offers pcap files manually(e.g., ISCX-IDS 2012, CIC-IDS 2017, BoT-IoT and ToN-IoT).

Here's brief pre-processing process that would be reference for you.

- You need to group packets(flow) in the pcap file with custom rules.
  - We defined packets with the same 5-tuple (source IP address, destination IP address, source port, destination port, transport layer protoco) as a flow.

- You need to label flows.
  - If your dataset doesn't offers flow labels, You cannot train this model.

- The flow consists of different number and size of packets. Since the DNN neet fixed size inputs, you have to fixed the size and number of packets consistuting a flow.
  - We defined the size of packet as 160 bytes.
  - We defined the number of packets as 10.

- You must remove or randomize the IP and MAC addresses that could be hint for label.
  - We simply randomize IP and MAC adresses.

- The shape of processed data is (batch size, number of packets, packet size).
  - If batch size is 100, the shape of batch data is (100, 10, 160).


# Reference
The work is an implementation of "Network Intrusion Detection: Based on Deep Hierarchical Network and Original Flow Data" (Zhang et al., 2019).

Paper: [https://ieeexplore.ieee.org/document/8672138]
