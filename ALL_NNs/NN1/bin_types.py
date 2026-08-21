# bin_types.py


"""
Bin type definitions and Bin class for 3D bin packing.
"""

# 
# BIN TYPES
# 

bin_types = \
[{'name': 'small',
  'length': 462,
  'width': 304,
  'height': 546,
  'cost': 10,
  'max_weight': 22},
 {'name': 'medium',
  'length': 745,
  'width': 325,
  'height': 399,
  'cost': 13,
  'max_weight': 14},
 {'name': 'large',
  'length': 635,
  'width': 441,
  'height': 413,
  'cost': 16,
  'max_weight': 26},
 {'name': 'extra_large',
  'length': 509,
  'width': 420,
  'height': 1260,
  'cost': 36,
  'max_weight': 15}]






# 
# BIN CLASS
# 

class Bin:
    """
    Represents a bin's dimensions and its available Empty Maximal Spaces (EMS)
    """
    def __init__(self, bin_type):
        self.name = bin_type["name"]
        self.length = bin_type["length"]
        self.width = bin_type["width"]
        self.height = bin_type["height"]
        self.cost = bin_type["cost"]

        # The first ems is the entire bin space
        self.ems_list = [(0, 0, 0, self.length, self.width, self.height)]

        # Keeping track of items placed inside
        self.placed_items = []
    
    def __repr__(self):
        return f"Bin({self.name}, EMSs: {len(self.ems_list)}, Items: {len(self.placed_items)})"