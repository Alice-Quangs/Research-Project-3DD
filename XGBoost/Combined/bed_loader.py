# bed_loader.py

import json

def load_bed_sequence(json_path="bed-bpp_v1.json", order_index=0):
    """
    Load one order sequence from BED-BPP dataset.
    Returns: list of dictionaries with item details
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get order IDs
    order_ids = list(data.keys())
    order_id = order_ids[order_index]
    order = data[order_id]
    
    # Extract items in sequence order
    sequence = []
    for key in sorted(order['item_sequence'].keys(), key=lambda x: int(x)):
        item = order['item_sequence'][key]
        sequence.append({
            'length': item['length/mm'],
            'width': item['width/mm'],
            'height': item['height/mm'],
            'weight': item['weight/kg'],
            'product_group': item['product_group'],
            'article': item['article'],
            'id': item['id']
        })
    
    return sequence


def num_orders(json_path="bed-bpp_v1.json"):
    """Small helper: how many orders are in the file, so callers can loop
    over every order_index without hardcoding a count."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)


def load_all_sequences(json_path="bed-bpp_v1.json"):
    """
    Load every order in the dataset and convert each item dict into the
    plain [length, width, height, weight] list format h.

    Reads and parses the JSON file exactly once 

    Returns: list of sequences, where each sequence is a list of
    [length, width, height, weight] lists.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    sequences = []
    for order_id, order in data.items():
        sequence = []
        for key in sorted(order['item_sequence'].keys(), key=lambda x: int(x)):
            item = order['item_sequence'][key]
            sequence.append([
                item['length/mm'],
                item['width/mm'],
                item['height/mm'],
                item['weight/kg'],
            ])
        sequences.append(sequence)

    return sequences