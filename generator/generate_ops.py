## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

import math
import csv
import os
import sys

class Tensor:
  def __init__(self, name : str, count : int, dims : list[int], dist : list[int], grid : list[int]):
    self.name = name
    self.count = count
    
    self.dims = dims
    self.dist = dist
    
    self.grid = grid
  
  def __get_distribution(self) -> str:
    distr = ""
  
    for i in range(len(self.dist)):
      valstr = '0' * len(self.grid)
      
      val = self.dist[i]
      
      if(val != 0):
        idx = (val - 1)
        vallist = list(valstr) 
        vallist[idx] = '1'
        valstr = "".join(vallist)

      if(i != len(self.dist) - 1):
        distr = distr + valstr + "x"
      else:
        distr = distr + valstr
    
    return distr

  def __get_distribution_last(self) -> str:
    distr = ""
  
    for i in range(len(self.dist) - 2, len(self.dist), 1):
      valstr = '0' * len(self.grid)
      
      val = self.dist[i]
      
      if(val != 0):
        idx = (val - 1)
        vallist = list(valstr) 
        vallist[idx] = '1'
        valstr = "".join(vallist)

      if(i != len(self.dist) - 1):
        distr = distr + valstr + "x"
      else:
        distr = distr + valstr
    
    return distr

  def get_local_size(self) -> float:
    dims = self.dims
    dist = self.dist
    
    local_size = 1
    for i in range(len(dims)):
      if(dist[i] == 0):
        p = 1
      else:
        p = self.grid[dist[i] - 1]
      
      local_size = local_size * int(dims[i] / p)
    
    return 16 * local_size / 1024.0 / 1024.0 / 1024.0
  
  def get_local_nr_elements(self) -> float:
    dims = self.dims
    dist = self.dist
    
    local_size = 1
    for i in range(len(dims)):
      if(dist[i] == 0):
        p = 1
      else:
        p = self.grid[dist[i] - 1]
      
      local_size = local_size * int(dims[i] / p)
    
    return local_size
  
  def get_split_local_dims(self, id_list : list[int]) -> list[int]:
    dims = self.dims
    dist = self.dist
    
    tx = []
    c = 1
    for i in range(len(dist)):
      if i not in id_list:
        if (dist[i] == 0):
          p = 1
        else:
          p = self.grid[dist[i] - 1]
        
        c = c * int(dims[i] / p)
      else:
        tx.append(c)
        
        if (dist[i] == 0):
          p = 1
        else:
          p = self.grid[dist[i] - 1]
          
        tx.append(int(dims[i] / p))
        
        c = 1
          
    tx.append(c)
    
    return tx

  def get_representation(self) -> str:
    if self.count == 0:
      return self.name + "[" + self.__get_distribution() + "]"
    else:
      return self.name + str(self.count) + "[" + self.__get_distribution() + "]"

  def get_representation_unique(self) -> str:
    if self.count == 0:
      return self.name + "_" + self.__get_distribution()
    else:
      return self.name + str(self.count) + "_" + self.__get_distribution()

  def get_representation_unique_last(self) -> str:
    if self.count == 0:
      return self.name + "_" + self.__get_distribution_last()
    else:
      return self.name + str(self.count) + "_" + self.__get_distribution_last()
  
class CommunicationDecomposition:
  def __init__(self, grid : list[int], ttype : int, threshold : int):
    self.threshold = threshold
    
    self.grid = grid
    self.transformations = []
    
    if ttype == 0:
      self.rules = [self.__breakdown_tensor_terminal, self.__breakdown_tensor_distribute_dim, self.__breakdown_tensor_replicate_dim, self.__breakdown_tensor_permute_dim, self.__breakdown_tensor_transpose_dims, self.__breakdown_tensor_swap_dims]
    elif ttype == 1:
      self.rules = [self.__breakdown_tensor_terminal, self.__breakdown_tensor_distribute_dim, self.__breakdown_tensor_replicate_dim, self.__breakdown_tensor_transpose_dims]
    elif ttype == 2:
      self.rules = [self.__breakdown_tensor_terminal, self.__breakdown_tensor_distribute_dim, self.__breakdown_tensor_replicate_dim]
    else:
      self.rules = [self.__breakdown_tensor_terminal, self.__breakdown_tensor_transpose_dims_general]

  def __check_uniqueness(self, dist : list[int]) -> bool:
    status = [0] * len(self.grid)
  
    for i in dist:
      if i != 0:
        status[i - 1] = status[i - 1] + 1
      
    for i in range(len(status)):
      if(status[i] > 1):
        return False
    
    return True

  # tensor decompositions
  def __breakdown_tensor_terminal(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    if(src_dist == dst_dist):
      return [[Transition("nop", src, dst, self.grid)]]
    
    return candidates
  
  def __breakdown_tensor_distribute_dim(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] == 0) and (dst_dist[i] != 0) and (dst_dist[i] not in src_dist):
        transformation = [i, -1, 0, -1, i, -1, dst_dist[i], -1]
        
        if transformation not in self.transformations:
          self.transformations.append(transformation)
          
          tmp_dims = src.dims.copy()
          tmp_dist = src.dist.copy()
          tmp_dist[i] = dst_dist[i]
          
          tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
          #child_l = Transition("distribute", src, tmp, self.grid)

          for rule in self.rules:
            children = rule(tmp, dst, depth + 1)
            
            for child_r in children:
              if(len(child_r) == 1) and child_r[0].name == 'nop':
                candidates.append([Transition("distribute", src, dst, self.grid)])
              else:
                candidates.append([Transition("distribute", src, tmp, self.grid)] + child_r)
              
          self.transformations.pop()
    
    return candidates

  def __breakdown_tensor_replicate_dim(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] != dst_dist[i]) and (src_dist[i] != 0):
        transformation = [i, -1, src_dist[i], -1, i, -1, 0, -1]
        
        if transformation not in self.transformations:
          self.transformations.append(transformation)
          
          tmp_dims = src.dims.copy()
          tmp_dist = src.dist.copy()
          tmp_dist[i] = 0
              
          tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid) 
          #child_l = Transition("replicate", src, tmp, self.grid)
          
          for rule in self.rules:
            children = rule(tmp, dst, depth + 1)
            
            for child_r in children:
              if(len(child_r) == 1) and child_r[0].name == 'nop':
                candidates.append([Transition("replicate", src, dst, self.grid)])
              else:
                candidates.append([Transition("replicate", src, tmp, self.grid)] + child_r)
              
          self.transformations.pop()
    
    return candidates

  def __breakdown_tensor_permute_dim(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] != 0) and (src_dist[i] != dst_dist[i]):
        for p in range(len(self.grid)):
          if(p + 1) not in src_dist:
            transformation0 = [i, -1, src_dist[i], -1, i, -1, p + 1, -1]
            transformation1 = [i, -1, p + 1, -1, i, -1, src_dist[i], -1]
            
            if transformation0 not in self.transformations and transformation1 not in self.transformations:
              self.transformations.append(transformation0)
                  
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = p + 1
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("permute", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("permute", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("permute", src, tmp, self.grid)] + child_r)    
                      
              self.transformations.pop()
    
    return candidates

  def __breakdown_tensor_transpose_dims(self, src, dst, depth):
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] != 0) and (src_dist[i] != dst_dist[i]):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] == 0) and (src_dist[j] != dst_dist[j]):
            transformation0 = [i, j, src_dist[i], src_dist[j], i, j, src_dist[j], src_dist[i]]
            transformation1 = [i, j, src_dist[j], src_dist[i], i, j, src_dist[i], src_dist[j]]
            
            if transformation0 not in self.transformations and transformation1 not in self.transformations:
              self.transformations.append(transformation0)
              
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = src_dist[j]
              tmp_dist[j] = src_dist[i]
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("transpose", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("transpose", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("transpose", src, tmp, self.grid)] + child_r)
                  
              self.transformations.pop()
              
    for i in range(len(src_dist)):
      if (src_dist[i] == 0) and (src_dist[i] != dst_dist[i]):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != 0) and (src_dist[j] != dst_dist[j]):
            transformation0 = [i, j, src_dist[i], src_dist[j], i, j, src_dist[j], src_dist[i]]
            transformation1 = [i, j, src_dist[j], src_dist[i], i, j, src_dist[i], src_dist[j]]
            
            if transformation0 not in self.transformations and transformation1 not in self.transformations:
              self.transformations.append(transformation0)
              
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = src_dist[j]
              tmp_dist[j] = src_dist[i]
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("transpose", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("transpose", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("transpose", src, tmp, self.grid)] + child_r)
                  
              self.transformations.pop()
    
    return candidates

  def __breakdown_tensor_transpose_dims_general(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] != 0):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] == 0):
            transformation0 = [i, j, src_dist[i], src_dist[j], i, j, src_dist[j], src_dist[i]]
            
            if transformation0 not in self.transformations:
              self.transformations.append(transformation0)
              
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = src_dist[j]
              tmp_dist[j] = src_dist[i]
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("transpose", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("transpose", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("transpose", src, tmp, self.grid)] + child_r)
                  
              self.transformations.pop()
    
    for i in range(len(src_dist)):
      if (src_dist[i] == 0):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != 0):
            transformation0 = [i, j, src_dist[i], src_dist[j], i, j, src_dist[j], src_dist[i]]
            
            if transformation0 not in self.transformations:
              self.transformations.append(transformation0)
              
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = src_dist[j]
              tmp_dist[j] = src_dist[i]
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("transpose", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("transpose", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("transpose", src, tmp, self.grid)] + child_r)
                  
              self.transformations.pop()
    
    return candidates

  def __breakdown_tensor_swap_dims(self, src : Tensor, dst : Tensor, depth : int) -> list:
    src_dist = src.dist
    dst_dist = dst.dist
    
    if(src.get_local_size() > 8):
      return []
    
    if(dst.get_local_size() > 8):
      return []
    
    if(depth >= self.threshold):
      return []
    
    candidates = []
    
    if(len(src_dist) != len(dst_dist)) or (self.__check_uniqueness(src_dist) == False) or (self.__check_uniqueness(dst_dist) == False):
      return candidates
    
    for i in range(len(src_dist)):
      if (src_dist[i] != 0) and (src_dist[i] != dst_dist[i]):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != 0) and (src_dist[j] != dst_dist[j]):
            transformation0 = [i, j, src_dist[i], src_dist[j], i, j, src_dist[j], src_dist[i]]
            transformation1 = [i, j, src_dist[j], src_dist[i], i, j, src_dist[i], src_dist[j]]
            
            if transformation0 not in self.transformations and transformation1 not in self.transformations:
              self.transformations.append(transformation0)
              
              tmp_dims = src.dims.copy()
              tmp_dist = src.dist.copy()
              tmp_dist[i] = src_dist[j]
              tmp_dist[j] = src_dist[i]
              
              tmp = Tensor(src.name, src.count + 1, tmp_dims, tmp_dist, self.grid)
              #child_l = Transition("swap", src, tmp, self.grid)
              
              for rule in self.rules:
                children = rule(tmp, dst, depth + 1)
                
                for child_r in children:
                  if(len(child_r) == 1) and child_r[0].name == 'nop':
                    candidates.append([Transition("swap", src, dst, self.grid)])
                  else:
                    candidates.append([Transition("swap", src, tmp, self.grid)] + child_r)
                  
              self.transformations.pop()
    
    return candidates
  
  def apply_rules(self, src : Tensor, dst : Tensor) -> list:  
    children = []
    for rule in self.rules:
      breakdowns = rule(src, dst, 0)

      for breakdown in breakdowns:
        children.append(TransitionChain(breakdown))
           
    return children

class Transition:
  def __init__(self, name : str, src : Tensor, dst : Tensor, grid : list[int]):
    self.name = name
    
    self.src = src
    self.dst = dst  
    
    self.grid = grid
  
    self.dimesions = []
  
    self.keys = []
    self.__generate_keys()
      
  def __generate_one_dimension(self, dist_i : int, dist_p : int) -> str:
    processors = [1] * len(self.grid)
    processors[dist_i] = dist_p
      
    total_processors = 1  
    processors_string = ""
      
    for i in range(len(self.grid)):
      total_processors = total_processors * self.grid[i]
        
      if (i < len(self.grid) - 1):
        processors_string = processors_string + str(processors[i]) + "_"
      else:
        processors_string = processors_string + str(processors[i])
      
    processors_string = str(total_processors) + "_" + processors_string
    
    return processors_string
  
  def __generate_two_dimensions(self, src_i : int, src_p : int, dst_i : int, dst_p : int) -> str:
    processors = [1] * len(self.grid)
    processors[src_i] = src_p
    processors[dst_i] = dst_p
      
    total_processors = 1  
    processors_string = ""
      
    for i in range(len(self.grid)):
      total_processors = total_processors * self.grid[i]
        
      if (i < len(self.grid) - 1):
        processors_string = processors_string + str(processors[i]) + "_"
      else:
        processors_string = processors_string + str(processors[i])
      
    processors_string = str(total_processors) + "_" + processors_string
    
    return processors_string
  
  def __generate_two_dimensions_smaller(self, src_i : int, src_p : int, dst_i0 : int, dst_p0 : int, dst_i1 : int, dst_p1 : int) -> list[str]:
    size = len(self.grid) + 1
    
    processors0 = [1] * size
    processors0[src_i] = src_p
    processors0[dst_i1] = dst_p1
      
    processors1 = [1] * size
    processors1[dst_i0] = dst_p0  
      
    total_processors = 1
    for i in range(len(self.grid)):
      total_processors = total_processors * self.grid[i]
     
    processors0_string = ""
    processors1_string = ""
    for i in range(size):
      if (i < size - 1):
        processors0_string = processors0_string + str(processors0[i]) + "_"
        processors1_string = processors1_string + str(processors1[i]) + "_"
      else:
        processors0_string = processors0_string + str(processors0[i])
        processors1_string = processors1_string + str(processors1[i])
      
    processors0_string = str(total_processors) + "_" + processors0_string
    processors1_string = str(total_processors) + "_" + processors1_string
    
    return [processors0_string, processors1_string]
  
  def __generate_two_dimensions_greater(self, src_i0 : int, src_p0 : int, src_i1 : int, src_p1 : int, dst_i : int, dst_p : int) -> list[str]:
    size = len(self.grid) + 1
    
    processors0 = [1] * (size)
    processors0[src_i1] = src_p1
    processors0[dst_i] = dst_p
    
    processors1 = [1] * (len(self.grid) + 1)
    processors1[src_i0] = src_p0
    
    total_processors = 1
    for i in range(len(self.grid)):
      total_processors = total_processors * self.grid[i]
     
    processors0_string = ""
    processors1_string = ""
    for i in range(size):
      if (i < size - 1):
        processors0_string = processors0_string + str(processors0[i]) + "_"
        processors1_string = processors1_string + str(processors1[i]) + "_"
      else:
        processors0_string = processors0_string + str(processors0[i])
        processors1_string = processors1_string + str(processors1[i])
      
    processors0_string = str(total_processors) + "_" + processors0_string
    processors1_string = str(total_processors) + "_" + processors1_string
    
    return [processors0_string, processors1_string]
  
  def __generate_distribute_cost(self):
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    idx = -1
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] == 0) and (dst_dist[i] != 0):
        idx = i
    
    if(idx != -1):
      [dst_l, dst_m, dst_r] = self.dst.get_split_local_dims([idx])
      dst_i = dst_dist[idx] - 1
      dst_p = self.grid[dst_i]

      if(dst_r != 1):
        self.dimesions.append(dst_l * dst_m * dst_p * dst_r)
        self.keys.append("pack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(dst_l, dst_m, dst_p, dst_r))
      
      self.dimesions.append(dst_l * dst_m * dst_p * dst_r)
      self.keys.append("filter_{0}_{1}_{2}".format(self.__generate_one_dimension(dst_i, dst_p), dst_l * dst_m * dst_r * dst_p, dst_l * dst_m * dst_r))
      self.dimesions.append(dst_l * dst_m * dst_r)

  def __generate_replicate_cost(self) -> float:
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    idx = -1
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] != 0) and (dst_dist[i] == 0):
        idx = i
    
    if(idx != -1):
      [src_l, src_m, src_r] = self.src.get_split_local_dims([idx])
      src_i = src_dist[idx] - 1
      src_p = self.grid[src_i]

      self.dimesions.append(src_l * src_m * src_r)
      self.keys.append("allgather_{0}_{1}_{2}".format(self.__generate_one_dimension(src_i, src_p), src_l * src_m * src_r, src_l * src_m * src_r * src_p))
      self.dimesions.append(src_l * src_m * src_r * src_p)
      if(src_r != 1):
        self.keys.append("unpack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l, src_m, src_r, src_p))
        self.dimesions.append(src_l * src_m * src_r * src_p)

  def __generate_permute_cost(self) -> float:
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    idx = -1
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] != 0) and (dst_dist[i] != 0):
        idx = i
    
    if(idx != -1):
      [src_l, src_m, src_r] = self.src.get_split_local_dims([idx])

      src_i = src_dist[idx] - 1
      src_p = self.grid[src_i]
      
      dst_i = dst_dist[idx] - 1
      dst_p = self.grid[dst_i]
      
      if(src_p == dst_p):
        self.dimesions.append(src_l * src_m * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(self.__generate_two_dimensions(src_i, src_p, dst_i, dst_p), src_l * src_m * src_r, src_l * src_m * src_r))
        self.dimesions.append(src_l * src_m * src_r)
      elif (src_p < dst_p):
        dst_p0 = int(dst_p / src_p)
        dst_p1 = src_p
        
        if(src_i < dst_i):
          src_i0 = src_i + 0
          dst_i0 = dst_i + 0
          dst_i1 = dst_i + 1
        
          prefix = self.__generate_two_dimensions_smaller(src_i, src_p, dst_i0, dst_p0, dst_i1, dst_p1)
        else:
          src_i0 = src_i + 1
          dst_i0 = dst_i + 0
          dst_i1 = dst_i + 1
          
          prefix = self.__generate_two_dimensions_greater(dst_i0, dst_p0, dst_i1, dst_p1, src_i0, src_p)
        
        if(src_r != 1):
          self.dimesions.append(src_l * src_m * src_r)
          self.keys.append("pack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l, int(src_m / dst_p0), dst_p0, src_r))
        self.dimesions.append(src_l * src_m * src_r)
        self.keys.append("filter_{0}_{1}_{2}".format(prefix[1], src_l * src_m * src_r, src_l * int(src_m / dst_p0) * src_r))
        self.dimesions.append(src_l * int(src_m / dst_p0) * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(prefix[0], src_l * int(src_m / dst_p0) * src_r, src_l * int(src_m / dst_p0) * src_r)) 
        self.dimesions.append(src_l * int(src_m / dst_p0) * src_r)       
      else:
        src_p0 = int(src_p / dst_p)  
        src_p1 = dst_p
        
        if(src_i < dst_i):
          src_i0 = src_i + 0
          src_i1 = src_i + 1
          dst_i0 = dst_i + 1
          
          prefix = self.__generate_two_dimensions_greater(src_i0, src_p0, src_i1, src_p1, dst_i0, dst_p)
        else:
          src_i0 = src_i + 0
          src_i1 = src_i + 1
          dst_i0 = dst_i + 0
          
          prefix = self.__generate_two_dimensions_smaller(dst_i0, dst_p, src_i0, src_p0, src_i1, src_p1)
        
        self.dimesions.append(src_l * src_m * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(prefix[0], src_l * src_m * src_r, src_l * src_m * src_r))
        self.dimesions.append(src_l * src_m * src_r)
        self.keys.append("allgather_{0}_{1}_{2}".format(prefix[1], src_l * src_m * src_r, src_l * src_m * src_r * src_p0))
        self.dimesions.append(src_l * src_m * src_r * src_p0)
        if(src_r != 1):
          self.keys.append("unpack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l, src_m, src_r, src_p0))
          self.dimesions.append(src_l * src_m * src_r * src_p0)

  def __generate_transpose_cost(self) -> float:
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    idx0 = -1
    idx1 = -1
    
    dist_i = -1
    dist_p = -1
    
    tx0 = [0, 0, 0]
    tx1 = [0, 0, 0]
    
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] != 0) and (dst_dist[i] == 0):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != dst_dist[j]) and (src_dist[j] == 0) and (dst_dist[j] != 0) and (src_dist[i] == dst_dist[j]):
            idx0 = i
            idx1 = j
            
            dist_i = src_dist[i] - 1
            dist_p = self.grid[dist_i]
            
            [src_l0, src_m0, src_m1, src_m2, src_r0] = self.src.get_split_local_dims([idx0, idx1])
            
            tx0[0] = src_l0 * src_m0 * src_m1
            tx0[1] = int(src_m2 / dist_p)
            tx0[2] = src_r0
            
            tx1[0] = src_l0
            tx1[1] = src_m0
            tx1[2] = src_m1 * int(src_m2 / dist_p) * src_r0

    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] == 0) and (dst_dist[i] != 0):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != dst_dist[j]) and (src_dist[j] != 0) and (dst_dist[j] == 0) and (dst_dist[i] == src_dist[j]):
            idx0 = i
            idx1 = j
            
            dist_i = dst_dist[i] - 1
            dist_p = self.grid[dist_i]
            
            [src_l0, src_m0, src_m1, src_m2, src_r0] = self.src.get_split_local_dims([idx0, idx1])
            
            tx0[0] = src_l0
            tx0[1] = int(src_m0 / dist_p)
            tx0[2] = src_m1 * src_m2 * src_r0
            
            tx1[0] = src_l0 * int(src_m0 / dist_p) * src_m1
            tx1[1] = src_m2
            tx1[2] = src_r0

    
    if(idx0 != -1 or idx1 != -1):
      size_l = tx0[0] * tx0[1] * tx0[2] * dist_p
      size_r = tx1[0] * tx1[1] * tx1[2] * dist_p
      
      if(tx0[2] != 1):
        self.dimesions.append(tx0[0] * tx0[1] * dist_p * tx0[2])
        self.keys.append("pack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(tx0[0], tx0[1], dist_p, tx0[2]))
      self.dimesions.append(tx0[0] * tx0[1] * dist_p * tx0[2])
      self.keys.append("alltoall_{0}_{1}_{2}".format(self.__generate_one_dimension(dist_i, dist_p), size_l, size_r))
      self.dimesions.append(tx1[0] * tx1[1] * dist_p * tx1[2])
      if(tx1[2] != 1):
        self.keys.append("unpack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(tx1[0], tx1[1], tx1[2], dist_p))
        self.dimesions.append(tx1[0] * tx1[1] * dist_p * tx1[2])
  
  def __generate_swap_cost(self) -> float:
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    idx0 = -1
    idx1 = -1    
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]) and (src_dist[i] != 0) and (dst_dist[i] != 0):
        for j in range(i + 1, len(src_dist), 1):
          if(src_dist[j] != dst_dist[j]) and (src_dist[j] != 0) and (dst_dist[j] != 0) and (src_dist[i] == dst_dist[j]) and (src_dist[j] == dst_dist[i]):
            idx0 = i
            idx1 = j

    if(idx0 != -1 or idx1 != -1):
      [src_l, src_m0, src_m1, src_m2, src_r] = self.src.get_split_local_dims([idx0, idx1])
      
      src_i = src_dist[idx0] - 1
      src_p = self.grid[src_i]
          
      dst_i = src_dist[idx1] - 1
      dst_p = self.grid[dst_i]
      
      if(src_p == dst_p):
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(self.__generate_two_dimensions(src_i, src_p, dst_i, dst_p), src_l * src_m0 * src_m1 * src_m2 * src_r, src_l * src_m0 * src_m1 * src_m2 * src_r))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
      elif (src_p < dst_p):
        dst_p0 = int(dst_p / src_p)
        dst_p1 = src_p
        
        if(src_i < dst_i):
          src_i0 = src_i + 0
          dst_i0 = dst_i + 0
          dst_i1 = dst_i + 1
        
          prefix = self.__generate_two_dimensions_smaller(src_i, src_p, dst_i0, dst_p0, dst_i1, dst_p1)
        else:
          src_i0 = src_i + 1
          dst_i0 = dst_i + 0
          dst_i1 = dst_i + 1
          
          prefix = self.__generate_two_dimensions_greater(dst_i0, dst_p0, dst_i1, dst_p1, src_i0, src_p)
        
        size_l = src_l * int(src_m0 / dst_p0) * src_m1 * src_m2 * src_r
        size_r = src_l * int(src_m0 / dst_p0) * src_m1 * src_m2 * src_r
        
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(prefix[0], src_l * src_m0 * src_m1 * src_m2 * src_r, src_l * src_m0 * src_m1 * src_m2 * src_r))
        if(src_m1 * src_m2 * src_r != 1):
          self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
          self.keys.append("pack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l, int(src_m0 / dst_p0), dst_p0, src_m1 * src_m2 * src_r))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        self.keys.append("alltoall_{0}_{1}_{2}".format(prefix[1], size_l, size_r))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        if(src_r != 1):
         self.keys.append("unpack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l * int(src_m0 / dst_p0) * src_m1, src_m2, src_r, dst_p0))
         self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
          
      else:
        src_p0 = int(src_p / dst_p)
        src_p1 = dst_p
        
        if(src_i < dst_i):
          src_i0 = src_i + 0
          src_i1 = src_i + 1
          dst_i0 = dst_i + 1
          
          prefix = self.__generate_two_dimensions_greater(src_i0, src_p0, src_i1, src_p1, dst_i0, dst_p)
        else:
          src_i0 = src_i + 0
          src_i1 = src_i + 1
          dst_i0 = dst_i + 0
          
          prefix = self.__generate_two_dimensions_smaller(dst_i0, dst_p, src_i0, src_p0, src_i1, src_p1)
        
        size_l = src_l * src_m0 * src_m1 * int(src_m2 / src_p0) * src_r
        size_r = src_l * src_m0 * src_m1 * int(src_m2 / src_p0) * src_r
        
        if(src_r != 1):
          self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
          self.keys.append("pack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l * src_m0 * src_m1, int(src_m2 / src_p0), src_p0, src_r))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        self.keys.append("alltoall_{0}_{1}_{2}".format(prefix[1], size_l, size_r))
        if(src_m1 * int(src_m2 / src_p0) * src_r != 1):
          self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
          self.keys.append("unpack_{0}x{1}x{2}x{3}_{0}x{1}x{3}x{2}".format(src_l, src_m0, src_m1 * int(src_m2 / src_p0) * src_r, src_p0))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
        self.keys.append("p2p_{0}_{1}_{2}".format(prefix[0], src_l * src_m0 * src_m1 * src_m2 * src_r, src_l * src_m0 * src_m1 * src_m2 * src_r))
        self.dimesions.append(src_l * src_m0 * src_m1 * src_m2 * src_r)
      
  def __generate_keys(self):
    src_dist = self.src.dist
    dst_dist = self.dst.dist
    
    if(len(src_dist) != (len(dst_dist))):
      print("Something is wrong")
      return
    
    if(self.name == "nop" and self.src.dist == self.dst.dist):
      self.dimesions = [self.src.get_local_nr_elements(), self.dst.get_local_nr_elements()]
      return
    
    one_diff_rules = [self.__generate_distribute_cost, self.__generate_replicate_cost, self.__generate_permute_cost]
    two_diff_rules = [self.__generate_transpose_cost, self.__generate_swap_cost]
  
    differences = 0  
    for i in range(len(src_dist)):
      if(src_dist[i] != dst_dist[i]):
        differences = differences + 1
    
    if (differences == 1):
      for rule in one_diff_rules:
        rule()

    if (differences == 2):
      for rule in two_diff_rules:
        rule()

  def get_representation(self) -> str:
    if self.src.dist != self.dst.dist:
      return self.dst.get_representation() + "=" + self.name + "(" + self.src.get_representation() + ")"
    return self.dst.get_representation() + "=" + self.src.get_representation()
  
  def get_dimensions(self):
    return self.dimesions
  
  def get_keys(self):
    op_keys = []
    
    if len(self.keys) == 1:
      op_keys = [self.keys[0] + "_" + self.src.name + str(self.src.count) + "_" + self.dst.name + str(self.dst.count)]
    else:
      for i in range(len(self.keys)):
        if i == 0:
          src_name = self.src.name + str(self.src.count)
          
          if "tx" in self.src.name:
            dst_name = self.src.name + str(self.src.count) + str(i)
          else:
            dst_name = "tx" + self.src.name + str(self.src.count) + str(i)
        elif i == len(self.keys) - 1:
          if "tx" in self.src.name:
            src_name = self.src.name + str(self.src.count) + str(i - 1)
          else:
            src_name = "tx" + self.src.name + str(self.src.count) + str(i - 1)
            
          dst_name = self.dst.name + str(self.dst.count)
        else:
          if "tx" in self.src.name:
            src_name = self.src.name + str(self.src.count) + str(i - 1)
            dst_name = self.src.name + str(self.src.count) + str(i)
          else:
            src_name = "tx" + self.src.name + str(self.src.count) + str(i - 1)
            dst_name = "tx" + self.src.name + str(self.src.count) + str(i)
        
        op_keys.append(self.keys[i] + "_" + src_name + "_" + dst_name)
    
    return op_keys
 
class TransitionChain:
  def __init__(self, operations : list):
    self.operations = operations
  
  def get_representation(self) -> str:
    operation_str = ""
    
    for op in self.operations:
      operation_str = operation_str + op.get_representation()
      
    return operation_str
  
  def get_number_of_operations(self):
    return len(self.operations)
  
  def get_dimensions(self):
    dimensions = []
    
    for op in self.operations:
      dimensions = dimensions + op.get_dimensions()
    
    return dimensions
  
  def get_keys(self):
    op_keys = []
    
    for op in self.operations:
      op_keys = op_keys + op.get_keys()
    
    return op_keys

class Computation:
  def __init__(self, name : str, operation : str, src_operands : list[Tensor], dst_operand : Tensor, appendix : str, grid : list[int]):
    self.name = name
    
    self.src_operands = src_operands
    self.dst_operand = dst_operand
    
    self.grid = grid
    
    self.dimensions = []
    
    for op in src_operands:
      self.dimensions.append(op.get_local_nr_elements())
    
    self.dimensions.append(dst_operand.get_local_nr_elements())
    
    self.operation = operation
    self.appendix = ""
      
    if(appendix != ""):
      self.appendix = appendix
      
  def get_representation(self) -> str:
    src_operands = ""
    
    for i in range(len(self.src_operands)):
      if(i < len(self.src_operands) - 1):
        src_operands = src_operands + self.src_operands[i].get_representation() + ", "
      else:
        src_operands = src_operands + self.src_operands[i].get_representation()
    
    return self.dst_operand.get_representation() + " = " + self.name + "(" + src_operands + ")" 
  
  def get_dimensions(self):
    return self.dimensions
  
  def change_output_type(self, rename, recount):
    self.dst_operand.name = rename
    self.dst_operand.count = recount
  
  def get_keys(self):
    input_operands = ""
    
    for i in range(len(self.src_operands)):
      if i < len(self.src_operands) - 1:
        input_operands = input_operands + self.src_operands[i].name + str(self.src_operands[i].count) + "_"
      else:
        input_operands = input_operands + self.src_operands[i].name + str(self.src_operands[i].count)
    
    if(self.appendix != ""):
      op_keys = [self.operation + "_" + input_operands + "_" + self.dst_operand.name + "x" + str(self.dst_operand.count)]
      op_keys = op_keys + [self.appendix + "_" + self.dst_operand.name + "x" + str(self.dst_operand.count) + "_" + self.dst_operand.name + str(self.dst_operand.count)]
    else:
      op_keys = [self.operation + "_" + input_operands + "_" + self.dst_operand.name + str(self.dst_operand.count)]
      
    return op_keys
    
class MatrixDecomposition:
  def __init__(self, grid : list[int], threshold : int):
    self.grid = grid
    
    self.communication_decomposition = CommunicationDecomposition(grid, 1, threshold)
    self.decompositions = []
    
  def clear_decompositions(self):
    self.decompositions.clear()  
  
  def __breakdown_x0_0y(self, x : int, y : int, mA : Tensor, mB : Tensor):
    if(mA.dims[1] != mB.dims[0]):
      return []
    
    m = mA.dims[0]
    k = mA.dims[1]
    n = mB.dims[1]
    
    if(x != 0):
      m = int(m / self.grid[x - 1])
      
    if(y != 0):
      n = int(n / self.grid[y - 1])
    
    computation_string = "gemm_{0}x{1}x{2}".format(m, k, n)
    
    if mA.dist == [x, 0]:
      opA = Tensor(mA.name, 0, mA.dims, [x, 0], self.grid)
    else:
      opA = Tensor(mA.name + "x", 0, mA.dims, [x, 0], self.grid)
      
    if mB.dist == [0, y]:
      opB = Tensor(mB.name, 0, mB.dims, [0, y], self.grid)
    else:
      opB = Tensor(mB.name + "x", 0, mB.dims, [0, y], self.grid)
      
    opC = Tensor("C", 0, [mA.dims[0], mB.dims[1]], [x, y], self.grid)
    
    if(opA.get_local_size() < 9.0 and opB.get_local_size() < 9.0):
      op_id = mA.get_representation_unique() + "_" + mB.get_representation_unique() + "_" + opC.get_representation_unique()
      computation = Computation("gemm", computation_string, [opA, opB], opC, "", self.grid)
      commsA = self.communication_decomposition.apply_rules(mA, opA)
      commsB = self.communication_decomposition.apply_rules(mB, opB)

      min_commA_val = commsA[0].get_number_of_operations()
      min_commA =  commsA[0]
      
      for commX in commsA:
        if commX.get_number_of_operations() < min_commA_val:
          min_commA_val = commX.get_number_of_operations()
          min_commA = commX
          
      min_commB_val = commsB[0].get_number_of_operations()
      min_commB =  commsB[0]
      
      for commX in commsB:
        if commX.get_number_of_operations() < min_commB_val:
          min_commB_val = commX.get_number_of_operations()
          min_commB = commX

      self.decompositions.append([op_id, computation,  [min_commA], [min_commB]])
    
  def __breakdown_yx_xz(self, x : int, y : int, z : int, mA : Tensor, mB : Tensor):
    if(mA.dims[1] != mB.dims[0]):
      return []
    
    if(x == 0) or (x == y) or (x == z):
      return []
    
    if((y != 0) or (z != 0)) and (y == z):
      return []
    
    m = mA.dims[0]
    k = mA.dims[1]
    n = mB.dims[1]
    
    processors = [1] * len(self.grid)
    
    if(y != 0):
      m = int(m / self.grid[y - 1])
    
    if(x != 0):
      k = int(k / self.grid[x - 1])
      processors[x - 1] = self.grid[x - 1]
      
    if(z != 0):
      n = int(n / self.grid[z - 1])
    
    total_size = 1
    processors_string = ""
    for i in range(len(self.grid)):
      total_size = total_size * self.grid[i]
      
      if(i < len(self.grid) - 1):
        processors_string = processors_string + str(processors[i]) + "_"    
      else:
        processors_string = processors_string + str(processors[i])
    processors_string = str(total_size) + "_" + processors_string
    
    computation_string = "gemm_{0}x{1}x{2}".format(m, k, n)
    
    if mA.dist == [y, x]:
      opA = Tensor(mA.name, 0, mA.dims, [y, x], self.grid)
    else:
      opA = Tensor(mA.name + "x", 0, mA.dims, [y, x], self.grid)
      
    if mB.dist == [x, z]:
      opB = Tensor(mB.name, 0, mB.dims, [x, z], self.grid)
    else:
      opB = Tensor(mB.name + "x", 0, mB.dims, [x, z], self.grid)
      
    opC = Tensor("C", 0, [mA.dims[0], mB.dims[1]], [y, z], self.grid)
    
    if(opA.get_local_size() < 9.0 and opB.get_local_size() < 9.0):
      op_id = mA.get_representation_unique() + "_" + mB.get_representation_unique() + "_" + opC.get_representation_unique()
      computation = Computation("gemm", computation_string, [opA, opB], opC, "allreduce_{0}_{1}_{1}".format(processors_string, m * n), self.grid)
      commsA = self.communication_decomposition.apply_rules(mA, opA)
      commsB = self.communication_decomposition.apply_rules(mB, opB)
      
      min_commA_val = commsA[0].get_number_of_operations()
      min_commA =  commsA[0]
      
      for commX in commsA:
        if commX.get_number_of_operations() < min_commA_val:
          min_commA_val = commX.get_number_of_operations()
          min_commA = commX
          
      min_commB_val = commsB[0].get_number_of_operations()
      min_commB =  commsB[0]
      
      for commX in commsB:
        if commX.get_number_of_operations() < min_commB_val:
          min_commB_val = commX.get_number_of_operations()
          min_commB = commX
      
      self.decompositions.append([op_id, computation, [min_commB], [min_commA]])
  
  def apply_rules(self, mA : Tensor, mB : Tensor):
    if len(mA.dist) != 2 or len(mB.dist) != 2:
      return
    
    [a, b] = mA.dist
    [c, d] = mB.dist
    
    if b == c:
      if b == 0:
        if a == d:
          if a == 0:
            self.__breakdown_x0_0y(0, 0, mA, mB)
          else:
            self.__breakdown_yx_xz(a, 0, 0, mA, mB)
        else:
          self.__breakdown_x0_0y(a, d, mA, mB)
      else:
        if a == d:
          self.__breakdown_yx_xz(b, a, 0, mA, mB)
        else:
          self.__breakdown_yx_xz(b, a, d, mA, mB)
    else:
      if b == 0:
        if a == c:
          self.__breakdown_yx_xz(c, 0, d, mA, mB)
        elif a == d:
          self.__breakdown_yx_xz(c, a, 0, mA, mB)
        else:
          self.__breakdown_yx_xz(c, a, d, mA, mB)
      elif c == 0:
        if b == d:
          self.__breakdown_yx_xz(b, a, 0, mA, mB)
        elif a == d:
          self.__breakdown_yx_xz(b, 0, d, mA, mB)
        else:
          self.__breakdown_yx_xz(b, a, d, mA, mB)
      else:
        if a == d:
          if a == 0:
            self.__breakdown_x0_0y(b, c, mA, mB)
          else:
            self.__breakdown_x0_0y(a, d, mA, mB)
        else:
          self.__breakdown_x0_0y(a, d, mA, mB)
        
  def get_keys(self):
    op_keys = []
    
    for decomposition in self.decompositions:
      for op in decomposition[2]:
        op_keys = op_keys + op.get_keys()
      
      for op in decomposition[3]:
        op_keys = op_keys + op.get_keys()
      
      op_keys = op_keys + decomposition[1].get_keys()

    return op_keys

class FourierDecomposition:
  def __init__(self, grid : list[int], threshold : int):
    self.communication_decomposition_lin = CommunicationDecomposition(grid, 1, threshold)
    
    self.grid = grid
    self.dimensions = []
    
    self.decompositions = []
  
  def clear_decompositions(self):
    self.decompositions.clear()  
  
  def __count_distributed_dimensions(self, dist : list[int]):
    count = 0
    for i in dist:
      if(i != 0):
        count = count + 1
    
    return count
  
  def __generate_steps(self, compute : list[int], tensor0 : Tensor, tensor1 : Tensor) -> list:
    dims = tensor0.dims
    dist = tensor0.dist
  
    computations = []
    
    total_nonzero = 0
    for id in range(len(compute)):
      if (compute[id] == 0):
        total_nonzero = total_nonzero + 1
    
    count = 0
    tmp0 = tensor0
    for id in range(len(compute)):        
      if(compute[id] == 0):       
        if count < total_nonzero - 1:
          tmp1 = Tensor("tx" + tmp0.name + "0", tmp0.count + 1, dims, dist, self.grid)
        else:
          tmp1 = tensor1 
        count = count + 1
          
        l = 1
        for i in range(0, id, 1):
          if(dist[i] == 0):
            p = 1
          else:
            p = self.grid[dist[i] - 1]
            
          l = l * int(dims[i] / p)
          
        r = 1
        for i in range(id + 1, len(dims), 1):
          if(dist[i] == 0):
            p = 1
          else:
            p = self.grid[dist[i] - 1]
            
          r = r * int(dims[i] / p)
        
        computations.append(Computation("fft", "fft_{0}x{1}x{2}".format(l, dims[id], r), [tmp0], tmp1, "", self.grid))
        
        tmp0 = tmp1

    return computations
  
  def __find_first_distribute_with_compute_right(self, compute : list[int]) -> list[int]:
    size = len(compute)
    for i0 in range(size):
      i1 = i0 + 1
      
      if(i1 > size - 1):
        i1 = i1 - size
      
      if (compute[i0] == 2) and (compute[i1] == 1):
        return [i0, i1]
    
    return [-1, -1]

  # for now
  def __find_first_distribute_with_compute_left(self, compute : list[int]) -> list[int]:
    size = len(compute)
    for i0 in range(size):
      i1 = i0 - 1
      
      if(i1 < 0):
        i1 = i1 + size
      
      if (compute[i0] == 2) and (compute[i1] == 1):
        return [i0, i1]
    
    return [-1, -1]
  
  def __breakdown_shift_nd(self, tA : Tensor, comp_steps : int, data_steps : int, func) -> list:    
    compute = [0] * comp_steps
    for id in range(comp_steps):
      if(tA.dist[id] != 0):
        compute[id] = 2
      else:
        compute[id] = 0
    
    tmp0 = Tensor("tx" + tA.name, tA.count + 1, tA.dims, tA.dist, self.grid)
    
    computations = self.__generate_steps(compute, tA, tmp0)
    for j in range(len(compute)):
      if compute[j] == 0:
        compute[j] = 1
    
    for i in range(data_steps):
      dims = tmp0.dims.copy()
      dist = tmp0.dist.copy()
      
      [id0, id1] = func(compute)
      compute[id0] = 0
      
      aux = dist[id0]
      dist[id0] = dist[id1]
      dist[id1] = aux
      
      tmp1 = Tensor(tmp0.name, tmp0.count + 2, dims, dist, self.grid)
      tmp2 = Tensor(tmp0.name, tmp0.count + 3, dims, dist, self.grid)
      
      trns = Transition("transpose", tmp0, tmp1, self.grid)
      computations.append(trns)
      
      computations = computations + self.__generate_steps(compute, tmp1, tmp2)
      for j in range(len(compute)):
        if compute[j] == 0:
          compute[j] = 1
 
      tmp0 = tmp2
    
    return [tmp0, computations]
  
  def apply_rules(self, tA : Tensor, tB : Tensor, type = 0):
    if(type != 0):
      if(tA.dist[-1] == 0 or tB.dist[-1] == 0 or tA.dist[-1] != tB.dist[-1]):
        return []
      
    if(tA.get_local_size() > 8):
      return []
    
    if(tB.get_local_size() > 8):
      return []
  
    if(type != 0):
      total_dimensions = len(tA.dist) - 1
      distributed_dimensions = self.__count_distributed_dimensions(tA.dist) - 1
    else:
      total_dimensions = len(tA.dist)
      distributed_dimensions = self.__count_distributed_dimensions(tA.dist)
      
    rules = [self.__find_first_distribute_with_compute_left]
    id = tA.get_representation_unique() + "_" + tB.get_representation_unique()
    
    for rule in rules:
      results = self.__breakdown_shift_nd(tA, total_dimensions, distributed_dimensions, rule)
      commsA = self.communication_decomposition_lin.apply_rules(results[0], tB)

      min_commA_val = commsA[0].get_number_of_operations()
      min_commA =  commsA[0]
      
      for commX in commsA:
        if commX.get_number_of_operations() < min_commA_val:
          min_commA_val = commX.get_number_of_operations()
          min_commA = commX
          
      if(len(min_commA.operations) == 1):
        if(min_commA.operations[0].name == "nop"):
          results[1][-1].change_output_type(tB.name, tB.count)

      self.decompositions.append([id, results[1], [min_commA]])
      
  def get_keys(self) -> list[str]:
    op_keys = []
    
    for decomposition in self.decompositions:
      for op in decomposition[1]:
        op_keys = op_keys + op.get_keys()
      
      for op in decomposition[2]:
        op_keys = op_keys + op.get_keys()
    
    return op_keys

def translate_comm(i, keys_i, max_sizes):
  key_elems = keys_i.split("_")
  
  temp_buffer = "temp"
  
  if not "tx" in key_elems[-2] and not "tx" in key_elems[-1]:
    src_buffer = key_elems[-2]
    dst_buffer = key_elems[-1]
  
  if not "tx" in key_elems[-2] and "tx" in key_elems[-1]:
    src_buffer = key_elems[-2]
    dst_buffer = temp_buffer + str((i + 0) % 2) 
    
  if "tx" in key_elems[-2] and "tx" in key_elems[-1]:
    src_buffer = temp_buffer + str((i - 1) % 2)
    dst_buffer = temp_buffer + str((i + 0) % 2) 
    
  if "tx" in key_elems[-2] and not "tx" in key_elems[-1]:
    src_buffer = temp_buffer + str((i - 1) % 2)
    dst_buffer = key_elems[-1]
  
  if 'pack' in keys_i and not 'unpack' in keys_i:
    key_elems = keys_i.split('_')
    pack_configs = key_elems[1].split('x')
    total_size = int(pack_configs[0]) * int(pack_configs[1]) * int(pack_configs[2]) * int(pack_configs[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    return ["pack_data({0}, {1}, {2}, {3}, (double*) {4}, {5}, (double*) {6});".format(int(pack_configs[0]) * int(pack_configs[1]), pack_configs[2], pack_configs[3], total_size, src_buffer, total_size, dst_buffer), max_sizes]
  
  if 'unpack' in keys_i:
    key_elems = keys_i.split('_')
    unpack_configs = key_elems[1].split('x')
    total_size = int(unpack_configs[0]) * int(unpack_configs[1]) * int(unpack_configs[2]) * int(unpack_configs[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    return ["unpack_data({0}, {1}, {2}, {3}, (double*) {4}, {5}, (double*) {6});".format(int(unpack_configs[0]) * int(unpack_configs[1]), unpack_configs[2], unpack_configs[3], total_size, src_buffer, total_size, dst_buffer), max_sizes]
  
  if 'alltoall' in keys_i:
    key_elems = keys_i.split('_')
    
    total_size = int(key_elems[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    if (len(key_elems) == 7):
      return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[3] + "/" + key_elems[2], src_buffer, key_elems[4] + "/" + key_elems[2], dst_buffer), max_sizes]
    elif (len(key_elems) == 8):
      if int(key_elems[2] != "1"):
        return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[4] + "/" + key_elems[2], src_buffer, key_elems[5] + "/" + key_elems[2], dst_buffer), max_sizes]
      else:
        return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm1", key_elems[4] + "/" + key_elems[3], src_buffer, key_elems[5] + "/" + key_elems[3], dst_buffer), max_sizes]
    elif (len(key_elems) == 9):
      if int(key_elems[2] != "1"):
        return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[5] + "/" + key_elems[2], src_buffer, key_elems[6] + "/" + key_elems[2], dst_buffer), max_sizes]
      elif int(key_elems[3] != "1"):
        return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm1", key_elems[5] + "/" + key_elems[3], src_buffer, key_elems[6] + "/" + key_elems[3], dst_buffer), max_sizes]
      else:
        return ["MPI_Alltoall((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm2", key_elems[5] + "/" + key_elems[4], src_buffer, key_elems[6] + "/" + key_elems[4], dst_buffer), max_sizes]
    else:
      print("Something is wrong")
      exit(-1)
      
  if 'allgather' in keys_i:
    key_elems = keys_i.split('_')
    
    total_size = int(key_elems[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    if (len(key_elems) == 7):
      return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[3], src_buffer, key_elems[4], dst_buffer), max_sizes]
    elif (len(key_elems) == 8):
      if int(key_elems[2] != "1"):
        return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[4], src_buffer, key_elems[5], dst_buffer), max_sizes]
      else:
        return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm1", key_elems[4], src_buffer, key_elems[5], dst_buffer), max_sizes]
    elif (len(key_elems) == 9):
      if int(key_elems[2] != "1"):
        return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm0", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
      elif int(key_elems[3] != "1"):
        return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm1", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
      else:
        return ["MPI_Allgather((double*) {2}, {1}, MPI_DOUBLE_COMPLEX, (double*) {4}, {3},  MPI_DOUBLE_COMPLEX, {0});".format("comm2", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
    else:
      print("Something is wrong")
      exit(-1)

  if 'filter' in keys_i:
    key_elems = keys_i.split('_')
    
    total_size = int(key_elems[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    if (len(key_elems) == 7):
      commands = "{\n"
      commands = commands + "\tint rank;\n"
      commands = commands + "MPI_Comm_rank(comm0, &rank);\n"
      commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
      commands = "}"
      
      return [commands, max_sizes]
    elif (len(key_elems) == 8):
      if int(key_elems[2] != "1"):
        commands = "{\n"
        commands = commands + "\tint rank;\n"
        commands = commands + "MPI_Comm_rank(comm0, &rank);\n"
        commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
        commands = "}"
      
        return [commands, max_sizes]
      else:
        commands = "{\n"
        commands = commands + "\tint rank;\n"
        commands = commands + "MPI_Comm_rank(comm1, &rank);\n"
        commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
        commands = "}"
        
        return [commands, max_sizes]
    elif (len(key_elems) == 9):
      if int(key_elems[2] != "1"):
        commands = "{\n"
        commands = commands + "\tint rank;\n"
        commands = commands + "MPI_Comm_rank(comm0, &rank);\n"
        commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
        commands = "}"
        
        return [commands, max_sizes]
      elif int(key_elems[3] != "1"):
        commands = "{\n"
        commands = commands + "\tint rank;\n"
        commands = commands + "MPI_Comm_rank(comm1, &rank);\n"
        commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
        commands = "}"
        
        return [commands, max_sizes]
      else:
        commands = "{\n"
        commands = commands + "\tint rank;\n"
        commands = commands + "MPI_Comm_rank(comm2, &rank);\n"
        commands = commands + "MPI_Filter((double*) {2}, {1}, (double*) {4}, {3}, {0});\n".format("rank", key_elems[3], src_buffer, key_elems[4], dst_buffer)
        commands = "}"
        
        return [commands, max_sizes]
    else:
      print("Something is wrong")
      exit(-1)
  
  if "allreduce" in keys_i:
    key_elems = keys_i.split('_')

    src_buffer = "temp0"
    dst_buffer = "C0"

    total_size = int(key_elems[3])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        
        if max_sizes[1] != 0:
          max_sizes[1] = a
      elif max_sizes[1] != 0 and total_size > max_sizes[1]:
        max_sizes[1] = total_size

    if (len(key_elems) == 7):
      return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm0", key_elems[3], src_buffer, key_elems[4], dst_buffer), max_sizes]
    elif (len(key_elems) == 8):
      if int(key_elems[2] != "1"):
        return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm0", key_elems[4], src_buffer, key_elems[5], dst_buffer), max_sizes]
      else:
        return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm1", key_elems[4], src_buffer, key_elems[5], dst_buffer), max_sizes]
    elif (len(key_elems) == 9):
      if int(key_elems[2] != "1"):
        return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm0", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
      elif int(key_elems[3] != "1"):
        return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm1", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
      else:
        return ["MPI_Allreduce((double*) {2}, (double*) {4}, {1}, MPI_DOUBLE_COMPLEX, MPI_SUM, {0});".format("comm2", key_elems[5], src_buffer, key_elems[6], dst_buffer), max_sizes]
    else:
      print("Something is wrong")
      exit(-1)

  return ["", max_sizes]

def translate_fft(i, keys_i, fft_type, fft_count, max_sizes):
  key_elems = keys_i.split("_")
  
  temp_buffer = "temp"
  
  if not "tx" in key_elems[-2] and "tx" in key_elems[-1]:
    src_buffer = key_elems[-2]
    dst_buffer = temp_buffer + str((i + 0) % 2) 
    
  if "tx" in key_elems[-2] and "tx" in key_elems[-1]:
    src_buffer = temp_buffer + str((i - 1) % 2)
    dst_buffer = temp_buffer + str((i + 0) % 2) 
    
  if "tx" in key_elems[-2] and not "tx" in key_elems[-1]:
    src_buffer = temp_buffer + str((i - 1) % 2)
    dst_buffer = key_elems[-1]
  
  if 'fft' in keys_i:
    key_elems = keys_i.split('_')
    fft_configs = key_elems[1].split('x')
    total_size = int(fft_configs[0]) * int(fft_configs[1]) * int(fft_configs[2])
    
    if "temp" in src_buffer:
      if total_size > max_sizes[0]:
        a = max_sizes[0]
        max_sizes[0] = total_size
        max_sizes[1] = a
      elif total_size > max_sizes[1]:
        max_sizes[1] = total_size
    
    if fft_type == 0:
      fft_type_value = "CUFFT_FORWARD"
    else:
      fft_type_value = "CUFFT_INVERSE"
    
    fft_count = fft_count + 1
    return ["fft_compute({8}, fft_plan{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7});".format(fft_count - 1, fft_configs[0], fft_configs[1], fft_configs[2], total_size, src_buffer, total_size, dst_buffer, fft_type_value), fft_count, max_sizes]
    
  [instructions, max_sizes] = translate_comm(i, keys_i, max_sizes)
  return [instructions, fft_count, max_sizes]
    
def translate_gemm(i, keys_i, max_sizes, aux_sizes):    
  if "gemm" in keys_i:
    key_elems = keys_i.split('_')
    gemm_configs = key_elems[1].split('x')
    
    size_input0 = int(gemm_configs[0]) * int(gemm_configs[1])
    size_input1 = int(gemm_configs[1]) * int(gemm_configs[2])
    
    size_output0 = int(gemm_configs[0]) * int(gemm_configs[2])

    src0_buffer = key_elems[2]
    src1_buffer = key_elems[3]
    
    if "x" in src0_buffer:
      aux_sizes[0] = size_input0
      
    if "x" in src1_buffer:
      aux_sizes[1] = size_input1

    if "x" in key_elems[4]:
      dst_buffer = "temp0"
    else:
      dst_buffer = "C0"

    return ["gemm_compute(gemm_plan, {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8});".format(gemm_configs[0], gemm_configs[1], gemm_configs[2], size_input0, src0_buffer, size_input1, src1_buffer, size_output0, dst_buffer), max_sizes, aux_sizes]

  [instructions, max_sizes] = translate_comm(i, keys_i, max_sizes)
  return [instructions, max_sizes, aux_sizes]

def generate_headers(op_type, f):
  print("#include <iostream>", file=f)
  print("#include <cstdlib>", file=f)
  print("#include <cstdio>", file=f)
  print("#include <mpi.h>", file=f)
  print("#include <complex>", file=f)
  print("#include <cuda_runtime.h>", file=f)
  print("#include <device_launch_parameters.h>", file=f)
  if(op_type == 0):
    print("#include <cufft.h>", file=f)
  elif (op_type == 1):
    print("#include <cublas_v2.h>", file=f)
  else:
    print("#include <cufft.h>", file=f)
    print("#include <cublas_v2.h>", file=f)
  print("", file=f)
  
  print("#include \"../templates/local_data.h\"", file=f)
  print("#include \"../templates/helper.h\"", file=f)
  print("", file=f)

def generate_fft_creation(instructions, f):
  value = 0
  for instruction in instructions:
    if "fft" in instruction:
      fft_split = instruction.split(",")
      
      print("\tcufftHandle fft_plan{0};".format(value), file=f)
      print("\tinx[0] = inembed[0] = onembed[0] = {0};".format(int(fft_split[3])), file=f)
      print("\tcufftPlanMany(&fft_plan{0}, 1, inx, inembed, 1, {1}, onembed, 1, {1}, CUFFT_Z2Z, {2});".format(value, fft_split[3], int(fft_split[2]) * int(fft_split[4])), file=f)
      print("", file=f)
      value = value + 1
        
def generate_fft_computation(f):
  print("__attribute__((always_inline)) inline void fft_compute(int fft_type, cufftHandle plan, int l, int m, int n, int size_in, std::complex<double> *input, int size_out, std::complex<double> *output)", file=f)
  print("{", file=f)
  print("\tstd::complex<double> *in_ptr = input;", file=f)
  print("\tstd::complex<double> *out_ptr = output;", file=f)
  print("", file=f)
  print("\tif(l != 1)", file=f)
  print("\t{", file=f)
  print("\t\ttranspose(l, m * n, (double*) in_ptr, (double*) out_ptr);", file=f)
  print("\t}", file=f)
  print("\telse", file=f)
  print("\t{", file=f)
  print("\t\tin_ptr = output;", file=f)
  print("\t\tout_ptr = input;", file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tcufftExecZ2Z(plan, (cufftDoubleComplex *)out_ptr, (cufftDoubleComplex *)in_ptr, fft_type);", file=f)
  print("", file=f)
  print("\tif(l != 1)", file=f)
  print("\t{", file=f)
  print("\t\ttranspose(m * n, l, (double*) in_ptr, (double*) out_ptr);", file=f)
  print("\t}", file=f)
  print("}", file=f)
  print("", file=f)

def generate_fft_destruction(size, f):
  print("\t// destroy fft_plans", file=f)
  for i in range(size):
    print("\tcufftDestroy(fft_plan{0});".format(i), file=f)
  print("", file=f)

def generate_gemm_creation(f):
  print("\tcublasHandle_t gemm_plan;", file=f)
  print("\tcublasCreate(&gemm_plan);", file=f)
  print("", file=f)
  
def generate_gemm_computation(f):
  print("__attribute__((always_inline)) inline void gemm_compute(cublasHandle_t handle, int m, int k, int n, int sizeA, std::complex<double> *A, int sizeB, std::complex<double> *B, int sizeC, std::complex<double> *C)", file=f)
  print("{", file=f)
  print("\tcuDoubleComplex alpha, beta;", file=f)
  print("\talpha.x = 1.0;", file=f)
  print("\talpha.y = 1.0;", file=f)
  print("\tbeta.x = 0.0;", file=f)
  print("\tbeta.y = 0.0;", file=f)
  print("", file=f)
  print("\tcublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (cuDoubleComplex *)B, n, (cuDoubleComplex *)A, k, &beta, (cuDoubleComplex *)C, n);", file=f)
  print("}", file=f)
  print("", file=f)

def generate_gemm_destruction(f):
  print("\tcublasDestroy(gemm_plan);", file=f)
  
def generate_grid(grid_shape, f):
  print("\t// create grid", file=f)
  if len(grid_shape) == 1:
    print("\tint rank;", file=f)
    print("\tMPI_Comm_rank(MPI_COMM_WORLD, &rank);", file=f)
    print("", file=f)
    print("\tMPI_Comm comm0 = MPI_COMM_WORLD;", file=f)
  elif len(grid_shape) == 2:
    print("\tint rank;", file=f)
    print("\tMPI_Comm_rank(MPI_COMM_WORLD, &rank);", file=f)
    print("", file=f)
    print("\tMPI_Comm comm0, comm1;", file=f)
    print("\tMPI_Comm_split(MPI_COMM_WORLD, rank / {0}, rank, &comm0);".format(grid_shape[0]), file=f)
    print("\tMPI_Comm_split(MPI_COMM_WORLD, rank % {0}, rank, &comm1);".format(grid_shape[0]), file=f)
  elif len(grid_shape) == 3:
    print("\tint rank;", file=f)
    print("\tMPI_Comm_rank(MPI_COMM_WORLD, &rank);", file=f)
    print("", file=f)
    print("\tMPI_Comm comm0, comm1, comm2;",file=f)
    print("\tMPI_Comm_split(MPI_COMM_WORLD, rank / {0}, rank, &comm0);".format(grid_shape[0]), file=f)
    print("\tMPI_Comm_split(MPI_COMM_WORLD, rank % {0} + {0} * (rank / ({0} * {1})), rank, &comm1);".format(grid_shape[0], grid_shape[1]), file=f)
    print("\tMPI_Comm_split(MPI_COMM_WORLD, rank % ({0} * {1}), rank, &comm2);".format(grid_shape[0], grid_shape[1]), file=f)
  else:
    print("Something went wrong")
    exit(-1)

def generate_create_counters(instructions, f):
  print("\t// create measuring parameters", file=f)
  print("\tcudaEvent_t counters[{0}];".format(len(instructions) + 1), file=f)
  print("\tfor(int c = 0; c < {0}; ++c)".format(len(instructions) + 1), file=f)
  print("\t{", file=f)
  print("\t\tcudaEventCreate(&counters[c]);", file=f)
  print("\t}", file=f) 
  print("", file=f)

def generate_measurements(instructions, f):
  print("\t\tcudaDeviceSynchronize();", file=f)
  print("\t\tMPI_Barrier(MPI_COMM_WORLD);", file=f)
  print("", file=f)
  print("\t\tif(r >= cold_runs)", file=f)
  print("\t\t{", file=f)
  print("\t\t\tfloat milliseconds;", file=f)
  print("", file=f)
  count = 0
  for instruction in instructions:
    print("\t\t\tcudaEventElapsedTime(&milliseconds, counters[{0}], counters[{1}]);".format(count + 0, count + 1), file=f)
    
    if "fft" in instruction or "gemm" in instruction:
      print("\t\t\tt_comp += milliseconds;", file=f)
    elif "pack" in instruction:
      print("\t\t\tt_pack += milliseconds;", file=f)
    else:
      print("\t\t\tt_comm += milliseconds;", file=f)
    
    count = count + 1

def generate_print_measurements(f, dimensions, direction, grid_shape):
  print("\t// get measuerments", file=f)
  print("\tMPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);", file=f)
  print("\tMPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);", file=f)
  print("\tMPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);", file=f)
  print("", file=f)
  
  dimension_description = ""
  for i in range(len(dimensions)):
    if i < len(dimensions) - 1:
      dimension_description = dimension_description + str(dimensions[i]) + " "
    else:
      dimension_description = dimension_description + str(dimensions[i])
  
  grid_description = ""
  for i in range(len(grid_shape)):
    if i < len(grid_shape) - 1:
      grid_description = grid_description + str(grid_shape[i]) + " "
    else:
      grid_description = grid_description + str(grid_shape[i])
  
  print("\tif(rank == 0)", file=f)
  
  if direction == "":
    print("\t\tprintf(\"{0}\\t{1}\\tComputation\\t%lf\\tPacking\\t%lf\\tCommunication\\t%lf\\tTotal\\t%lf\\n\", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);".format(dimension_description, grid_description), file=f)
  else:
    print("\t\tprintf(\"{0}\\t{1}\\t{2}\\tComputation\\t%lf\\tPacking\\t%lf\\tCommunication\\t%lf\\tTotal\\t%lf\\n\", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);".format(dimension_description, direction, grid_description), file=f)
    
  print("", file=f)

def generate_destroy_measurements(instructions, f):
  print("\t// destroy measuring parameters", file=f)
  print("\tfor(int c = 0; c < {0}; ++c)".format(len(instructions) + 1), file=f)
  print("\t{", file=f)
  print("\t\tcudaEventDestroy(counters[c]);", file=f)
  print("\t}", file=f) 
  print("", file=f)

def generate_main_function(f):
  print("int main(int argc, char **argv)", file=f)
  print("{", file=f)
  print("\tMPI_Init(&argc, &argv);", file=f)
  print("\tint cold_runs = atoi(argv[1]);", file=f)
  print("\tint hot_runs = atoi(argv[2]);", file=f)
  print("\tfunction(cold_runs, hot_runs);", file=f)
  print("\tMPI_Finalize();", file=f)
  print("\treturn 0;", file=f)
  print("}", file=f)

# general Fourier transforms
def fft_generation(filename, direction, fft_dim_size, A, B, instructions, total_temp_buffers, max_temp_buffer_size, grid_shape):
  f = open(filename, "w")
  
  distIn = A.dist
  distOut = B.dist
  
  generate_headers(0, f)
  generate_fft_computation(f)
  
  print("// ", end='\t', file=f)
  print(distIn, end='\t', file=f)
  print(distOut, file=f)
  print("void function(int cold_runs, int hot_runs)", file=f)
  print("{", file=f)
  
  generate_grid(grid_shape, f)
  
  print("", file=f)
  print("\t// create arrays", file=f)
  print("\tstd::complex<double> *{0}_host = NULL, *{1}_host = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tdouble real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tdouble imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tstd::complex<double> value(real_part, imag_part);", file=f)
  print("\t\t*({0}_host + i) = value;".format(A.name + str(A.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tstd::complex<double> value(0.0, 0.0);", file=f)
  print("\t\t*({0}_host + i) = value;".format(B.name + str(B.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\t// create fft_plans", file=f)
  print("\tint inx[1], inembed[1], onembed[1];", file=f)
  print("", file=f)
  
  generate_fft_creation(instructions, f)
  
  print("\t// create device arrays", file=f)
  print("\tstd::complex<double> *{0} = NULL, *{1} = NULL, *temp = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  
  if(total_temp_buffers != 0):
    print("\tcudaMalloc((void**)&temp, {0} * sizeof(std::complex<double>));\t".format(total_temp_buffers * max_temp_buffer_size), file=f)
    
    if total_temp_buffers == 1:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
    else:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
      print("\tstd::complex<double> *temp1 = (temp + 1 * {0});".format(max_temp_buffer_size), file=f)
  
  print("", file=f)
  print("\t// copy data from arrays to device arrays", file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("", file=f)
  
  generate_create_counters(instructions, f)
  
  print("\tdouble t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;", file=f)
  print("\tfor(int r = 0; r < cold_runs + hot_runs; ++r)", file=f)
  print("\t{", file=f)   
  print("\t\t// add measuring things", file=f)
  print("\t\tcudaEventRecord(counters[0]);", file=f)
  count = 1
  for instruction in instructions:
    print("\t\t" + instruction, file=f)
    print("\t\t// add measuring things", file=f)
    print("\t\tcudaEventRecord(counters[{0}]);".format(count), file=f)
    count = count + 1
  print("", file=f)
  
  generate_measurements(instructions, f)
  
  print("\t\t}", file=f)
  print("\t}", file=f)
  print("\t// copy data from device arrays to device", file=f)
  print("\tcudaMemcpy({0}_host, {0}, {1} * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("", file=f)
  
  generate_print_measurements(f, A.dims, direction, grid_shape)
  
  generate_fft_destruction(fft_dim_size, f)
  
  print("\t// destroy dev arrays", file=f)
  print("\tcudaFree({0});".format(A.name + str(A.count)), file=f)
  print("\tcudaFree({0});".format(B.name + str(B.count)), file=f)
  if(total_temp_buffers != 0):
    print("\tcudaFree(temp);", file=f)
  print("", file=f)
  
  generate_destroy_measurements(instructions, f)
  
  print("\t// destroy arrays", file=f)
  print("\tfree({0}_host);".format(A.name + str(A.count)), file=f)
  print("\tfree({0}_host);".format(B.name + str(B.count)), file=f)
  print("}", file=f)
  print("", file=f)
  
  generate_main_function(f)
  
  f.close()

def fft_d_call(filename, batched, fft_sizes, distIn, distOut, grid_shape):
  fwd = FourierDecomposition(grid_shape, 4)
  
  fft_type = 0
  if batched == 1:
    fft_dim_size = len(fft_sizes) - 1
  else:
    fft_dim_size = len(fft_sizes)
  
  A = Tensor("A", 0, fft_sizes, distIn, grid_shape)
  B = Tensor("B", 0, fft_sizes, distOut, grid_shape)
  
  fwd.apply_rules(A, B, batched)
  keys = fwd.get_keys()

  if(len(keys) == 0):
    return False

  instructions = []
  
  fft_count = 0
  max_sizes = [0, 0]
  for i in range(len(keys)): 
    [instruction, fft_count, max_sizes] = translate_fft(i, keys[i], fft_type, fft_count, max_sizes) 
    instructions.append(instruction)
  
  max_temp_buffer_size = max(max_sizes)
  total_temp_buffers = sum([x != 0 for x in max_sizes])
  
  fft_generation(filename, "D", fft_dim_size, A, B, instructions, total_temp_buffers, max_temp_buffer_size, grid_shape)
  return True

def fft_di_call(filename, batched, fft_sizes, distIn, distMiddle, distOut, grid_shape):
  fwd = FourierDecomposition(grid_shape, 4)
  bwd = FourierDecomposition(grid_shape, 4)
  
  fft_type_forward = 0
  fft_type_inverse = 1
  
  if batched == 1:
    fft_dim_size = 2 * (len(fft_sizes) - 1)
  else:
    fft_dim_size = 2 * (len(fft_sizes))
    
  A = Tensor("A", 0, fft_sizes, distIn, grid_shape)
  B = Tensor("Btx", 0, fft_sizes, distMiddle, grid_shape)
  C = Tensor("C", 0, fft_sizes, distOut, grid_shape)
  
  fwd.apply_rules(A, B, batched)
  keys_forward = fwd.get_keys()
  
  if(len(keys_forward) == 0):
    return False
  
  bwd.apply_rules(B, C, batched)
  keys_inverse = bwd.get_keys()
  
  if(len(keys_inverse) == 0):
    return False

  instructions = []
  
  fft_count = 0
  max_sizes = [0, 0]
  for i in range(len(keys_forward)): 
    [instruction, fft_count, max_sizes] = translate_fft(i, keys_forward[i], fft_type_forward, fft_count, max_sizes) 
    instructions.append(instruction)
    
  offset = len(instructions)
  for i in range(len(keys_inverse)): 
    [instruction, fft_count, max_sizes] = translate_fft(i + offset, keys_inverse[i], fft_type_inverse, fft_count, max_sizes) 
    instructions.append(instruction)
  
  max_temp_buffer_size = max(max_sizes)
  total_temp_buffers = sum([x != 0 for x in max_sizes])
  
  fft_generation(filename, "DI", fft_dim_size, A, C, instructions, total_temp_buffers, max_temp_buffer_size, grid_shape)
  return True

# general matrix multiplication
def matrix_generation(filename, m, n, A, B, instructions, total_temp_buffers, max_temp_buffer_size, total_aux_buffers, total_aux_buffer_size, aux_sizes, grid_shape):
  f = open(filename, "w")
  
  distA = A.dist
  distB = B.dist
  
  generate_headers(1, f)
  
  generate_gemm_computation(f)
  
  print("// ", end='\t', file=f)
  print(distA, end='\t', file=f)
  print(distB, file=f)
  print("void function(int cold_runs, int hot_runs)", file=f)
  print("{", file=f)
  
  generate_grid(grid_shape, f)
  
  print("", file=f)
  print("\t// create arrays", file=f)
  print("\tstd::complex<double> *{0}_host = NULL, *{1}_host = NULL, *C0_host = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tC0_host = (std::complex<double>*) malloc({0} * sizeof(std::complex<double>));\t".format(m * n), file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tdouble real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tdouble imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tstd::complex<double> value(real_part, imag_part);", file=f)
  print("\t\t*({0}_host + i) = value;".format(A.name + str(A.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tdouble real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tdouble imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tstd::complex<double> value(real_part, imag_part);", file=f)
  print("\t\t*({0}_host + i) = value;".format(B.name + str(B.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(m * n), file=f)
  print("\t{", file=f)
  print("\t\tstd::complex<double> value(0.0, 0.0);", file=f)
  print("\t\t*(C0_host + i) = value;", file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\t// create gemm_plans", file=f)
  
  generate_gemm_creation(f)
  
  print("\t// create device arrays", file=f)
  print("\tstd::complex<double> *{0} = NULL, *{1} = NULL, *C0, *temp = NULL, *aux = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMalloc((void**)&C0, {0} * sizeof(std::complex<double>));\t".format(m * n), file=f)
  
  if(total_temp_buffers != 0):
    print("\tcudaMalloc((void**)&temp, {0} * sizeof(std::complex<double>));\t".format(total_temp_buffers * max_temp_buffer_size), file=f)
    
    if total_temp_buffers == 1:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
    else:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
      print("\tstd::complex<double> *temp1 = (temp + 1 * {0});".format(max_temp_buffer_size), file=f)
  
  if(total_aux_buffers != 0):
    print("\tcudaMalloc((void**)&aux, {0} * sizeof(std::complex<double>));\t".format(total_aux_buffer_size), file=f)
    
    if aux_sizes[0] != 0 and aux_sizes[1] == 0:
      print("\tstd::complex<double>* Ax0 = temp;", file=f)
    elif aux_sizes[0] == 0 and aux_sizes[1] != 0:
      print("\tstd::complex<double>* Bx0 = temp;", file=f)
    elif aux_sizes[0] != 0 and aux_sizes[1] != 0:
      print("\tstd::complex<double>* Ax0 = (temp + 0);", file=f)
      print("\tstd::complex<double>* Bx0 = (temp + {0});".format(aux_sizes[0]), file=f)
    else:
      print("Something is wrong")
      exit(-1)
    
  print("", file=f)
  print("\t// copy data from arrays to device arrays", file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(B.name + str(B.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("", file=f)
  
  generate_create_counters(instructions, f)
  
  print("\tdouble t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;", file=f)
  print("\tfor(int r = 0; r < cold_runs + hot_runs; ++r)", file=f)
  print("\t{", file=f)   
  print("\t\t// add measuring things", file=f)
  print("\t\tcudaEventRecord(counters[0]);", file=f)
  count = 1
  for instruction in instructions:
    print("\t\t" + instruction, file=f)
    print("\t\t// add measuring things", file=f)
    print("\t\tcudaEventRecord(counters[{0}]);".format(count), file=f)
    count = count + 1
  print("", file=f)
  
  generate_measurements(instructions, f)
  
  print("\t\t}", file=f)
  print("\t}", file=f)
  print("\t// copy data from device arrays to device", file=f)
  print("\tcudaMemcpy(C0_host, C0, {0} * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);\t".format(m * n), file=f)
  print("", file=f)
  
  generate_print_measurements(f, A.dims, "", grid_shape)
  
  generate_gemm_destruction(f)
  
  print("\t// destroy dev arrays", file=f)
  print("\tcudaFree(A0);", file=f)
  print("\tcudaFree(B0);", file=f)
  print("\tcudaFree(C0);", file=f)
  if(total_temp_buffers != 0):
    print("\tcudaFree(temp);", file=f)
    
  if(total_aux_buffers != 0):
    print("\tcudaFree(aux);", file=f)
    
  print("", file=f)
  
  generate_destroy_measurements(instructions, f)
  
  print("\t// destroy arrays", file=f)
  print("\tfree({0}_host);".format(A.name + str(A.count)), file=f)
  print("\tfree({0}_host);".format(B.name + str(B.count)), file=f)
  print("\tfree(C0_host);", file=f)
  print("}", file=f)
  print("", file=f)
  
  generate_main_function(f)
  
  f.close()

def mm(filename, matrix_sizes, distA, distB, grid_shape):
  md = MatrixDecomposition(grid_shape, 4)
    
  m = matrix_sizes[0]
  k = matrix_sizes[1]
  n = matrix_sizes[2]
  
  A = Tensor("A", 0, [m, k], distA, grid_shape)
  B = Tensor("B", 0, [k, n], distB, grid_shape)
  
  md.apply_rules(A, B)
  keys = md.get_keys()    
  
  if(len(keys) == 0):
    return False
  
  instructions = []
  
  max_sizes = [0, 0]
  aux_sizes = [0, 0]
  for i in range(len(keys)): 
    [instruction, max_sizes, aux_sizes] = translate_gemm(i, keys[i], max_sizes, aux_sizes)
    
    instructions.append(instruction)
  
  max_temp_buffer_size = max(max_sizes)
  total_temp_buffers = sum([x != 0 for x in max_sizes])
  
  total_aux_buffer_size = sum(aux_sizes)
  total_aux_buffers = sum([x != 0 for x in aux_sizes])
  
  matrix_generation(filename, m, n, A, B, instructions, total_temp_buffers, max_temp_buffer_size, total_aux_buffers, total_aux_buffer_size, aux_sizes, grid_shape)
  return True

# general matrix multiplication
def fft_matrix_generation(filename, direction, fft_dim_size, m, n, A, B, instructions, total_temp_buffers, max_temp_buffer_size, total_aux_buffers, total_aux_buffer_size, aux_sizes, grid_shape):
  f = open(filename, "w")
  
  distA = A.dist
  distB = B.dist
  
  generate_headers(2, f)
  
  generate_fft_computation(f)
  generate_gemm_computation(f)
  
  print("// ", end='\t', file=f)
  print(distA, end='\t', file=f)
  print(distB, file=f)
  print("void function(int cold_runs, int hot_runs)", file=f)
  print("{", file=f)
  
  generate_grid(grid_shape, f)
  
  print("", file=f)
  print("\t// create arrays", file=f)
  print("\tstd::complex<double> *{0}_host = NULL, *{1}_host = NULL, *C0_host = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{0}_host = (std::complex<double>*) malloc({1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tC0_host = (std::complex<double>*) malloc({0} * sizeof(std::complex<double>));\t".format(m * n), file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tdouble real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tdouble imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tstd::complex<double> value(real_part, imag_part);", file=f)
  print("\t\t*({0}_host + i) = value;".format(A.name + str(A.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\t{", file=f)
  print("\t\tdouble real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tdouble imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);", file=f)
  print("\t\tstd::complex<double> value(real_part, imag_part);", file=f)
  print("\t\t*({0}_host + i) = value;".format(B.name + str(B.count)), file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\tfor(int i = 0; i < {0}; ++i)".format(m * n), file=f)
  print("\t{", file=f)
  print("\t\tstd::complex<double> value(0.0, 0.0);", file=f)
  print("\t\t*(C0_host + i) = value;", file=f)
  print("\t}", file=f)
  print("", file=f)
  print("\t// create fft_plans", file=f)
  print("\tint inx[1], inembed[1], onembed[1];", file=f)
  print("", file=f)
  
  generate_fft_creation(instructions, f)
  
  print("\t// create gemm_plans", file=f)
  
  generate_gemm_creation(f)
  
  print("\t// create device arrays", file=f)
  print("\tstd::complex<double> *{0} = NULL, *{1} = NULL, *C0, *temp = NULL, *aux = NULL;".format(A.name + str(A.count), B.name + str(B.count)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMalloc((void**)&{0}, {1} * sizeof(std::complex<double>));\t".format(B.name + str(B.count), int(B.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMalloc((void**)&C0, {0} * sizeof(std::complex<double>));\t".format(m * n), file=f)
  
  if(total_temp_buffers != 0):
    print("\tcudaMalloc((void**)&temp, {0} * sizeof(std::complex<double>));\t".format(total_temp_buffers * max_temp_buffer_size), file=f)
    
    if total_temp_buffers == 1:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
    else:
      print("\tstd::complex<double> *temp0 = (temp + 0 * {0});".format(max_temp_buffer_size), file=f)
      print("\tstd::complex<double> *temp1 = (temp + 1 * {0});".format(max_temp_buffer_size), file=f)
  
  if(total_aux_buffers != 0):
    print("\tcudaMalloc((void**)&aux, {0} * sizeof(std::complex<double>));\t".format(total_aux_buffer_size), file=f)
    
    if aux_sizes[0] != 0 and aux_sizes[1] == 0:
      print("\tstd::complex<double>* Ax0 = temp;", file=f)
    elif aux_sizes[0] == 0 and aux_sizes[1] != 0:
      print("\tstd::complex<double>* Tx0 = temp;", file=f)
    elif aux_sizes[0] != 0 and aux_sizes[1] != 0:
      print("\tstd::complex<double>* Ax0 = (temp + 0);", file=f)
      print("\tstd::complex<double>* Tx0 = (temp + {0});".format(aux_sizes[0]), file=f)
    else:
      print("Something is wrong")
      exit(-1)
    
  print("", file=f)
  print("\t// copy data from arrays to device arrays", file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(A.name + str(A.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("\tcudaMemcpy({0}, {0}_host, {1} * sizeof(std::complex<double>), cudaMemcpyHostToDevice);\t".format(B.name + str(B.count), int(A.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)), file=f)
  print("", file=f)
  
  generate_create_counters(instructions, f)
  
  print("\tdouble t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;", file=f)
  print("\tfor(int r = 0; r < cold_runs + hot_runs; ++r)", file=f)
  print("\t{", file=f)   
  print("\t\t// add measuring things", file=f)
  print("\t\tcudaEventRecord(counters[0]);", file=f)
  count = 1
  for instruction in instructions:
    print("\t\t" + instruction, file=f)
    print("\t\t// add measuring things", file=f)
    print("\t\tcudaEventRecord(counters[{0}]);".format(count), file=f)
    count = count + 1
  print("", file=f)
  
  generate_measurements(instructions, f)
  
  print("\t\t}", file=f)
  print("\t}", file=f)
  print("\t// copy data from device arrays to device", file=f)
  print("\tcudaMemcpy(C0_host, C0, {0} * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);\t".format(m * n), file=f)
  print("", file=f)
  
  generate_print_measurements(f, B.dims, direction, grid_shape)
  
  generate_fft_destruction(fft_dim_size, f)
  
  generate_gemm_destruction(f)
  
  print("\t// destroy dev arrays", file=f)
  print("\tcudaFree(A0);", file=f)
  print("\tcudaFree(B0);", file=f)
  print("\tcudaFree(C0);", file=f)
  if(total_temp_buffers != 0):
    print("\tcudaFree(temp);", file=f)
    
  if(total_aux_buffers != 0):
    print("\tcudaFree(aux);", file=f)
    
  print("", file=f)
  
  generate_destroy_measurements(instructions, f)
  
  print("\t// destroy arrays", file=f)
  print("\tfree({0}_host);".format(A.name + str(A.count)), file=f)
  print("\tfree({0}_host);".format(B.name + str(B.count)), file=f)
  print("\tfree(C0_host);", file=f)
  print("}", file=f)
  print("", file=f)
  
  generate_main_function(f)
  
  f.close()

def fft_d_mm_call(filename, fft_sizes, distIn, distOut, distA, distB, grid_shape):
  #part of the linearizer
  distL = distOut
  distR = ([0] * (len(distOut) - len(distB))) + distB
  
  # the Fourier transform
  fwd = FourierDecomposition(grid_shape, 4)
  
  fft_type = 0
  fft_dim_size = len(fft_sizes) - 1

  B = Tensor("B", 0, fft_sizes, distIn, grid_shape)
  if(distL != distR):
    Tf = Tensor("TLtx", 0, fft_sizes, distOut, grid_shape)
  else:
    Tf = Tensor("T", 0, fft_sizes, distOut, grid_shape)
  
  fwd.apply_rules(B, Tf, 1)
  keys = fwd.get_keys()

  if(len(keys) == 0):
    return False

  fft_instructions = []
  
  fft_count = 0
  max_sizes = [0, 0]
  for i in range(len(keys)): 
    [fft_instruction, fft_count, max_sizes] = translate_fft(i, keys[i], fft_type, fft_count, max_sizes) 
    fft_instructions.append(fft_instruction)
  
  # the linearizer
  if(distL != distR):
    cd = CommunicationDecomposition(grid_shape, 1, 4)
    
    TL = Tensor("TLtx", 0, fft_sizes, distL, grid_shape)
    TR = Tensor("T", 0, fft_sizes, distR, grid_shape)
    
    comms = cd.apply_rules(TL, TR)
    
    min_comm_val = comms[0].get_number_of_operations()
    min_comm =  comms[0]
    
    for commX in comms:
      if commX.get_number_of_operations() < min_comm_val:
        min_comm_val = commX.get_number_of_operations()
        min_comm = commX
        
    offset = len(fft_instructions)
    
    keys = min_comm.get_keys()
    for i in range(len(keys)): 
      [comm_instruction, max_sizes] = translate_comm(i + offset, keys[i], max_sizes)
      fft_instructions.append(comm_instruction)
  
  # the gemm part
  m = fft_sizes[-1]
  k = math.prod(fft_sizes[:-1])
  n = fft_sizes[-1]
  
  md = MatrixDecomposition(grid_shape, 4)
  
  A = Tensor("A", 0, [m, k], distA, grid_shape)
  Tm = Tensor("T", 0, [k, n], distB, grid_shape)
  
  md.apply_rules(A, Tm)
  keys = md.get_keys()    
  
  if(len(keys) == 0):
    return False
  
  gemm_instructions = []
  aux_sizes = [0, 0]
  for i in range(len(keys)): 
    [gemm_instruction, max_sizes, aux_sizes] = translate_gemm(i + len(fft_instructions), keys[i], max_sizes, aux_sizes)
    gemm_instructions.append(gemm_instruction)
  
  if aux_sizes[1] == 0:
    name = Tm.name + str(Tm.count)
    rename = Tm.name + "x" + str(Tm.count)
  
    aux_sizes[1] = int(Tm.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)
  else:
    name = Tm.name + str(Tm.count)
    rename = "temp" + str((len(fft_instructions) + 1) % 2)
    
    total_size = int(Tm.get_local_size() / 16 * 1024 * 1024 * 1024)
    
    if total_size > max_sizes[0]:
      a = max_sizes[0]
      max_sizes[0] = total_size
      max_sizes[1] = a
    elif total_size > max_sizes[1]:
      max_sizes[1] = total_size
  
  instructions = []
  for instruction in fft_instructions:
    instruction = instruction.replace(name, rename)
    instructions.append(instruction)
    
  for instruction in gemm_instructions:
    instruction = instruction.replace(name, rename)
    instructions.append(instruction)
    
  max_temp_buffer_size = max(max_sizes)
  total_temp_buffers = sum([x != 0 for x in max_sizes])
  
  total_aux_buffer_size = sum(aux_sizes)
  total_aux_buffers = sum([x != 0 for x in aux_sizes])
  
  A_matrix = Tensor("A", 0, [m, k], distA, grid_shape)
  B_fft = Tensor("B", 0, fft_sizes, distIn, grid_shape)
  
  fft_matrix_generation(filename, "D", fft_dim_size, m, n, A_matrix, B_fft, instructions, total_temp_buffers, max_temp_buffer_size, total_aux_buffers, total_aux_buffer_size, aux_sizes, grid_shape)
  return True

def fft_di_mm_call(filename, fft_sizes, distIn, distMiddle, distOut, distA, distB, grid_shape):
  #part of the linearizer
  distL = distOut
  distR = ([0] * (len(distOut) - len(distB))) + distB
  
  # the Fourier transform
  fwd = FourierDecomposition(grid_shape, 4)
  bwd = FourierDecomposition(grid_shape, 4)
  
  fft_type_forward = 0
  fft_type_inverse = 1

  fft_dim_size = 2 * (len(fft_sizes) - 1)
    
  B = Tensor("B", 0, fft_sizes, distIn, grid_shape)
  M = Tensor("Mtx", 0, fft_sizes, distMiddle, grid_shape)
  if(distL != distR):
    Tf = Tensor("TLtx", 0, fft_sizes, distOut, grid_shape)
  else:
    Tf = Tensor("T", 0, fft_sizes, distOut, grid_shape)
  
  fwd.apply_rules(B, M, 1)
  keys_forward = fwd.get_keys()
  
  if(len(keys_forward) == 0):
    return False
  
  bwd.apply_rules(M, Tf, 1)
  keys_inverse = bwd.get_keys()
  
  if(len(keys_inverse) == 0):
    return False

  fft_instructions = []
  
  fft_count = 0
  max_sizes = [0, 0]
  for i in range(len(keys_forward)): 
    [instruction, fft_count, max_sizes] = translate_fft(i, keys_forward[i], fft_type_forward, fft_count, max_sizes) 
    fft_instructions.append(instruction)
  
  offset = len(fft_instructions)
  for i in range(len(keys_inverse)): 
    [instruction, fft_count, max_sizes] = translate_fft(i + offset, keys_inverse[i], fft_type_inverse, fft_count, max_sizes) 
    fft_instructions.append(instruction)
  
  # the linearizer
  if(distL != distR):
    cd = CommunicationDecomposition(grid_shape, 1, 4)
    
    TL = Tensor("TLtx", 0, fft_sizes, distL, grid_shape)
    TR = Tensor("T", 0, fft_sizes, distR, grid_shape)
    
    comms = cd.apply_rules(TL, TR)
    
    min_comm_val = comms[0].get_number_of_operations()
    min_comm =  comms[0]
    
    for commX in comms:
      if commX.get_number_of_operations() < min_comm_val:
        min_comm_val = commX.get_number_of_operations()
        min_comm = commX
        
    offset = len(fft_instructions)
    
    keys = min_comm.get_keys()
    for i in range(len(keys)): 
      [comm_instruction, max_sizes] = translate_comm(i + offset, keys[i], max_sizes)
      fft_instructions.append(comm_instruction)
  
  # the gemm part
  m = fft_sizes[-1]
  k = math.prod(fft_sizes[:-1])
  n = fft_sizes[-1]
  
  md = MatrixDecomposition(grid_shape, 4)
  
  A = Tensor("A", 0, [m, k], distA, grid_shape)
  Tm = Tensor("T", 0, [k, n], distB, grid_shape)
  
  md.apply_rules(A, Tm)
  keys = md.get_keys()    
  
  if(len(keys) == 0):
    return False
  
  gemm_instructions = []
  aux_sizes = [0, 0]
  for i in range(len(keys)): 
    [gemm_instruction, max_sizes, aux_sizes] = translate_gemm(i + len(fft_instructions), keys[i], max_sizes, aux_sizes)
    gemm_instructions.append(gemm_instruction)
  
  if aux_sizes[1] == 0:
    name = Tm.name + str(Tm.count)
    rename = Tm.name + "x" + str(Tm.count)
  
    aux_sizes[1] = int(Tm.get_local_size() / 16 * 1024.0 * 1024.0 * 1024.0)
  else:
    name = Tm.name + str(Tm.count)
    rename = "temp" + str((len(fft_instructions) + 1) % 2)
    
    total_size = int(Tm.get_local_size() / 16 * 1024 * 1024 * 1024)
    
    if total_size > max_sizes[0]:
      a = max_sizes[0]
      max_sizes[0] = total_size
      max_sizes[1] = a
    elif total_size > max_sizes[1]:
      max_sizes[1] = total_size
  
  instructions = []
  for instruction in fft_instructions:
    instruction = instruction.replace(name, rename)
    instructions.append(instruction)
    
  for instruction in gemm_instructions:
    instruction = instruction.replace(name, rename)
    instructions.append(instruction)
    
  max_temp_buffer_size = max(max_sizes)
  total_temp_buffers = sum([x != 0 for x in max_sizes])
  
  total_aux_buffer_size = sum(aux_sizes)
  total_aux_buffers = sum([x != 0 for x in aux_sizes])
  
  A_matrix = Tensor("A", 0, [m, k], distA, grid_shape)
  B_fft = Tensor("B", 0, fft_sizes, distIn, grid_shape)
  
  fft_matrix_generation(filename, "DI", fft_dim_size, m, n, A_matrix, B_fft, instructions, total_temp_buffers, max_temp_buffer_size, total_aux_buffers, total_aux_buffer_size, aux_sizes, grid_shape)
  return True

# the next couple of functions deal with the 2D FFT  
def fft2d_d(filename, batched, fft_sizes, distIn, distOut, grid_shape):
  if batched == 1:
    lower_bound = 2
    upper_bound = 3
  else:
    lower_bound = 1
    upper_bound = 2
  
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 2")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
  
  return fft_d_call(filename, batched, fft_sizes, distIn, distOut, grid_shape)

def fft2d_di(filename, batched, fft_sizes, distIn, distMiddle, distOut, grid_shape):
  if batched == 1:
    lower_bound = 2
    upper_bound = 3
  else:
    lower_bound = 1
    upper_bound = 2
    
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 2")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)

  return fft_di_call(filename, batched, fft_sizes, distIn, distMiddle, distOut, grid_shape)

# the next couple of functions deal with the 3D FFT
def fft3d_d(filename, batched, fft_sizes, distIn, distOut, grid_shape):
  if batched == 1:
    lower_bound = 2
    upper_bound = 4
  else:
    lower_bound = 1
    upper_bound = 3
  
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 3")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
  
  return fft_d_call(filename, batched, fft_sizes, distIn, distOut, grid_shape)

def fft3d_di(filename, batched, fft_sizes, distIn, distMiddle, distOut, grid_shape):
  if batched == 1:
    lower_bound = 2
    upper_bound = 4
  else:
    lower_bound = 1
    upper_bound = 3
    
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 3")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
    
  return fft_di_call(filename, batched, fft_sizes, distIn, distMiddle, distOut, grid_shape)

# the next couple of functions deal twith the batched 2D FFT and gemm
def fft2d_d_b_mm(filename, fft_sizes, distIn, distOut, distA, distB, grid_shape):
  lower_bound = 2
  upper_bound = 3
  
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 2")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
  
  return fft_d_mm_call(filename, fft_sizes, distIn, distOut, distA, distB, grid_shape)

def fft2d_di_b_mm(filename, fft_sizes, distIn, distMiddle, distOut, distA, distB, grid_shape):
  lower_bound = 2
  upper_bound = 3
  
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 2")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
  
  return fft_di_mm_call(filename, fft_sizes, distIn, distMiddle, distOut, distA, distB, grid_shape)

# the next couple of functions deal twith the batched 3D FFT and gemm
def fft3d_di_b_mm(filename, fft_sizes, distIn, distMiddle, distOut, distA, distB, grid_shape):
  lower_bound = 2
  upper_bound = 4
  
  if (len(fft_sizes) != upper_bound):
    print("This should be a direct DFT of size 2")
    exit(-1)
  
  if not ((len(grid_shape) >= lower_bound) and (len(grid_shape) < len(fft_sizes))):
    print("The grid has to be smaller than the DFT dimension size")
    exit(-1)
  
  return fft_di_mm_call(filename, fft_sizes, distIn, distMiddle, distOut, distA, distB, grid_shape)

def can_convert_to_int(value):
  try:
    int(value)
    return True
  except (ValueError, TypeError):
    return False

def generate_code(filepath):
  directory = os.path.dirname(filepath)
  
  with open(filepath, "r") as file:
    reader = csv.reader(file, delimiter='|') 
        
    for e in reader:
      values = e[1].split("_")
      
      grid_shape = []
      tensor_shapes = []
      
      count = 0
      while count < len(values):
        if(can_convert_to_int(values[count])):
          grid_shape.append(int(values[count]))
        else:
          if "x" not in values[count]:
            distributions = values[count + 1].split("x")
            
            distribution_values = []
            for distribution in distributions:
              distribution_values.append(distribution.find('1') + 1)
            
            tensor_shapes.append(distribution_values)
            count = count + 1
        
        count = count + 1
      
      e_split = e[0].split("_")
      
      if "_fb_" in e[0]:
        direction = "di"
      else:
        direction = "d"
      
      fft_name = ""
      fft_dim_sizes = []
      
      matrix_name = ""
      
      if "fft2D" in e[0]:
        if "batch" in e[0]:
          fft_name = fft_name + "fft_" + direction + "_" + e_split[-3] + "_" + e_split[-2] + "_" + e_split[-1] + "_"
          
          if "gemm" in e[0]:
            matrix_name = matrix_name + "mm_" + e_split[-1] + "_" + str(int(e_split[-3]) * int(e_split[-2])) + "_" + e_split[-1] + "_"
            
          fft_dim_sizes.append(int(e_split[-3]))
          fft_dim_sizes.append(int(e_split[-2]))
          fft_dim_sizes.append(int(e_split[-1]))
        else:
          fft_name = fft_name + "fft_" + direction + "_" + e_split[-2] + "_" + e_split[-1] + "_"
          fft_dim_sizes.append(int(e_split[-2]))
          fft_dim_sizes.append(int(e_split[-1]))

      if "fft3D" in e[0]:
        if "batch" in e[0]:
          fft_name = fft_name + "fft_" + direction + "_" + e_split[-4] + "_" + e_split[-3] + "_" + e_split[-2] + "_" + e_split[-1] + "_"
          
          if "gemm" in e[0]:
            matrix_name = matrix_name + "mm_" + e_split[-1] + "_" + str(int(e_split[-4]) * int(e_split[-3]) * int(e_split[-2])) + "_" + e_split[-1] + "_"
            
          fft_dim_sizes.append(int(e_split[-4]))
          fft_dim_sizes.append(int(e_split[-3]))
          fft_dim_sizes.append(int(e_split[-2]))
          fft_dim_sizes.append(int(e_split[-1]))
        else:
          fft_name = fft_name + "fft_" + direction + "_" + e_split[-3] + "_" + e_split[-2] + "_" + e_split[-1] + "_"
          fft_dim_sizes.append(int(e_split[-3]))
          fft_dim_sizes.append(int(e_split[-2]))
          fft_dim_sizes.append(int(e_split[-1]))
      
      grid_name = "grid_" + str(len(grid_shape)) + "D"
      filename = directory + "/" + fft_name + matrix_name + "proc_" + str(math.prod(grid_shape)) + "_" + grid_name + ".cpp"
      
      print(filename)
      
      if "gemm" in e[0]:
        if "fb" in e[0]:
          if len(fft_dim_sizes) == 3:
            print(fft2d_di_b_mm(filename, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], tensor_shapes[2], tensor_shapes[3], tensor_shapes[4], grid_shape))
          else: 
            print(fft3d_di_b_mm(filename, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], tensor_shapes[2], tensor_shapes[3], tensor_shapes[4], grid_shape))
        else:
          if len(fft_dim_sizes) == 3:
            print(fft2d_d_b_mm(filename, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], tensor_shapes[2], tensor_shapes[3], grid_shape))
          else:
            print("Not implemented")
      else:
        dimensions = 2
        batched = 0
        if "batch" in e[0]:
          dimensions = 3
          batched = 1
        
        if "fb" in e[0]:
          if len(fft_dim_sizes) == dimensions:
            print(fft2d_di(filename, batched, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], tensor_shapes[2], grid_shape))
          else:
            print(fft3d_di(filename, batched, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], tensor_shapes[2], grid_shape))
        else:
          if len(fft_dim_sizes) == dimensions:
            print(fft2d_d(filename, batched, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], grid_shape))
          else:
            print(fft3d_d(filename, batched, fft_dim_sizes, tensor_shapes[0], tensor_shapes[1], grid_shape))
            
if __name__ == "__main__":
  if (sys.version_info[0] < 3 or sys.version_info[1] < 10):
    print("Requering a python version that is greater than 3.10")
  
  fft_list = ["./fft_2D", "./fft_2D_batch", "./fft_2D_batch_mm", "./fft_3D", "./fft_3D_batch", "./fft_3D_batch_mm"]
  
  for option in fft_list:
    generate_code(option + "/results.csv")