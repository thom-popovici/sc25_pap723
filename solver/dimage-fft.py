## DiMage-DFT: The DIstributed MApping GEnerator for Distributed Fourier Transforms.
## Authors: Doru Popovici, Botao Wu, John Shalf, Martin Kong.
## Maintainer: Martin Kong.
## Copyright 2025. Ohio State University.

import sys 
import re
import os
import math
import signal
from timeit import default_timer as timer
from fractions import gcd
#import fftspace
import datetime
from decimal import Decimal

DIMAGE_MAX_TRIES=1
DIMAGE_ONE_SHOT=False
DIMAGE_DEBUG=False
DO_REF=True
DIMAGE_PY_SCRIPT="dimage.py"
MEM2COMP_RATIO = 40
## Tweak variable below to accommodate possible internal local memory used.
DIMAGE_CAP_FACTOR = 1
DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY=False
DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS=True
DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS=False
DIMAGE_FAST_SOL=False
DIMAGE_OPTION_DO_CHECK=True
DIMAGE_USE_RHO_REPLICATION_FACTOR=False
DIMAGE_OPTION_DEBUG=0   # Higher is more verbose.
DIMAGE_ERROR_FILE='dimage.error'
DIMAGE_ERROR_COUNT=0

# Operator types
DIMAGE_OP_GENERATOR=0
DIMAGE_OP_SINK=1

# Capacity-related constants.
DIMAGE_DEFAULT_MAXCAP=(128 * 2**30)

ROT_UNREACH=9999
DIMAGE_DFT_ROT_COMPLEMENT_COST=5

PER_DIM = 1
PER_PROC = 2
USE_MODULO = False
DIM_UNMAPPED = -1
DIM_NOT_USED=-2
DIMAGE_INT = 'int'
DIMAGE_DT = 'complex'  #'double'
DIMAGE_COMPUTE_COLOR_FUNC = 'dimage_compute_color_from_comm_vec'
DIMAGE_GRID_DIMS = 'DIMAGE_GRID_DIMS'
DIMAGE_TILE_HEADER_SIZE=8
DIMAGE_TILE_HEADER_MACRO='DIMAGE_TILE_HEADER_SIZE'
DIMAGE_BUFFER_ALLOCATOR='dimage_alloc_buffer'
DIMAGE_TILE_ALLOCATOR_1D='dimage_1d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_2D='dimage_2d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_3D='dimage_3d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_4D='dimage_4d_tile_alloc'
DIMAGE_TILE_MAP_ALLOCATOR='dimage_alloc_tile_map'
DIMAGE_COLLECT_TILE_MAP_FUNC='dimage_collect_tile_coordinates'
DIMAGE_STORE_TILE_MAP_FUNC='dimage_store_tile_map'
DIMAGE_FETCH_TILE_FUNC='dimage_fetch_tile_ptr'
DIMAGE_SET_TILE_COORD_FUNC='DIMAGE_SET_TILE_COORDINATE'
ALLOC_MODE_FULL=0
ALLOC_MODE_SLICE=1
ALLOC_MODE_TILE=2
DIMAGE_CEIL_DEF='#define dimage_ceil(n,d)  ceil((((double)(n))*(DIMAGE_SF))/((double)(d)))'
DIMAGE_CEIL='aceil'
DIMAGE_SF_DEF="#define DIMAGE_SF 1"
DIMAGE_SF='DIMAGE_SF'
BASE_INDENT='  '
COMM_TYPE_LOCAL=0
COMM_TYPE_LOCAL_SLICE=1
COMM_TYPE_GATHER_SLICE=2
COMM_TYPE_ALLRED=3
COMM_TYPE_P2P=4
COLLECTIVE_ALLGATHER='MPI_Allgather'
COLLECTIVE_ALLREDUCE='MPI_Allreduce'
DIMAGE_PROC_COORD_FUNC='dimage_rank_to_coords'
ACC_TYPE_TILE=0
ACC_TYPE_SLICE=1
ACC_TYPE_LIN=2
ACC_TYPE_ERROR=-42
L2_LOOP_GENMODE_FULL=0
L2_LOOP_GENMODE_LB=1
L2_LOOP_GENMODE_UB=2
DIMAGE_KERNEL_FUNCALL='dimage_gemm'
DIMAGE_ACC_LIN='DIMAGE_ACC_LIN'
DIMAGE_ACC_TILE='DIMAGE_ACC_TILE'
DIMAGE_ACC_SLICE='DIMAGE_ACC_SLICE'
DIMAGE_TILE_POINTER='DIMAGE_PTR_TILE'
WRITE_TO_FILE_FUNC='write_to_file'
READ_FROM_FILE_FUNC='read_from_file'
WRITE_MATRIX_TO_FILE='write_to_file_matrix'
ARRAY_CHECK_FUNC='check_array'
DIMAGE_INIT_DIAG='DIMAGE_INIT_DIAG'
REDUCE_OP_ADD='MPI_SUM'
DIMAGE_PROC_RANK='dimage_rank'
DIMAGE_RANK_ARRAY='dimage_ranks'
DIMAGE_PROC_COORDS='dimage_coords'
DIMAGE_CLOCK='rtclock()'
DIMAGE_START_TIMER='timer_start'
COMM_SIZE_VAR='procs_in_comm'
DIMAGE_BLOCK_COUNT='block_count'
DIMAGE_REFBLOCK_COUNT='ref_block_count'
MPI_COMM_SIZE='MPI_Comm_size'
DIMAGE_TIMEOUT=-1
DIMAGE_OBJ_COMM_ONLY=1
DIMAGE_OBJ_COMM_COMP=2
DIMAGE_CHECK_NO_CHECK=0
DIMAGE_CHECK_READ_REF_ARRAY=1
DIMAGE_CHECK_CALL_CHECK=2
DEBUG_BLOCK_SIZE_OP_TEN_DIM=False
DEBUG_REF_USED_DIM=False

def timeout_message (signum, frame):
  print ("Solver timed-out ({} seconds)".format (DIMAGE_TIMEOUT))
  raise exception ("[DIMAGE:TIMEOUT]")

def gen_error_file (oper_name, msg):
  global DIMAGE_ERROR_COUNT
  fmode = 'w'
  filename = DIMAGE_ERROR_FILE 
  if (DIMAGE_ERROR_COUNT >= 1):
    fmode = 'a'
  ff = open (filename, fmode)
  DIMAGE_ERROR_COUNT += 1
  info = '{}, erro.count= {}'.format (oper_name, DIMAGE_ERROR_COUNT)
  ff.write (msg + '\n')
  ff.write (info + '\n')
  ff.write ('\n***********************************************\n\n')
  ff.close ()

def prod (ll):
  ret = 1
  for xx in ll:
    ret = ret * xx
  return ret

# Return False if the per_node capacity is 0, 0K, 0k, 0M or 0m,
# and True otherwise.
def include_capacity_constraints (per_node_cap):
  pnc = re.sub ('[KkMm]','', per_node_cap)
  if (pnc.find ("0") == 0):
    return False
  return True

def iceil(num,den):
  return int(math.ceil(num/(1.0*den)))

def comm_type_str(ct):
  if (ct == COMM_TYPE_LOCAL):
    return 'LOCAL'
  if (ct == COMM_TYPE_ALLRED):
    return 'ALLRED'
  if (ct == COMM_TYPE_LOCAL_SLICE):
    return 'LOCAL_SLICE'
  if (ct == COMM_TYPE_GATHER_SLICE):
    return 'GATHER_SLICE'
  if (ct == COMM_TYPE_P2P):
    return 'P2P'
  return 'ERROR'

def get_mpi_datatype (dtype):
  if (dtype == 'double'):
    return 'MPI_DOUBLE'
  if (dtype == 'float'):
    return 'MPI_FLOAT'
  if (dtype == 'int'):
    return 'MPI_INT'
  return 'ERROR'

def build_sum_constraint (varlist):
  ret = ''
  for vv in varlist:
    if (ret != ''):
      ret += ' + '
    ret += str(vv)
  return ret

def dimage_timestamp ():
  return datetime.datetime.now()

def print_current_time ():
  msg=dimage_timestamp ()
  print ('Current time: {}'.format (msg))

def estimate_per_node_requirement (scol, PP, procs):
  max_cap = 0
  caps = []
  for ss in scol:
    stmt = scol[ss]
    caps.append (stmt.estimate_memory_requirements ())
  max_req = max(caps)
  if (DIMAGE_DT == 'complex'):
    max_req *= 16
  if (DIMAGE_DT == 'double'):
    max_req *= 8
  if (DIMAGE_DT == 'float'):
    max_req *= 4
  gsize = 1
  unit = 'B'
  if (max_req >= 2**30):
    max_req = float(max_req) / 3**20
    unit = 'GB'
  elif (max_req >= 2**20):
    max_req = float(max_req) / 2**20
    unit = 'MB'
  elif (max_req >= 2**10):
    max_req = float(max_req) / 2**10
    unit = 'KB'
  print   ("Single-node requirement : {} {}".format (max_req, unit ))
  if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
    for pp in procs:
      gsize *= pp
      req_per_node = int(math.ceil (max_req/gsize))
      print ("Requirement by {} nodes : {} {}".format (gsize, req_per_node, unit))
  else:
    n_proc_dim = PP.get_num_dim ()
    max_procs = PP.get_max_procs ()
    proc_per_dim = max_procs ** (1.0 / n_proc_dim)
    for pp in range(n_proc_dim):
      nodes_level = proc_per_dim ** (pp + 1)
      req_per_node = max_req * 1.0 / nodes_level
      print ("Requirement at level {} - {} nodes: {} {}".format (pp + 1, nodes_level, req_per_node, unit))



## Z3 optimization flags and optimization options.
class Comm_Opt_Form:
  def __init__ (self, output_filename, procvec):
    self.decl = []
    self.cstr = []
    self.modelfile = output_filename
    self.pvec = procvec
    self.options = ""
    self.options += "':algebraic_number_evaluator', False, "
    self.options += "':arith_ineq_lhs', False, " 
    self.options += "':eq2ineq', False, " 
    self.options += "':expand_nested_stores', True, "
    self.options += "':gcd_rounding', False, "
    self.options += "':ignore_patterns_on_ground_qbody', True, "
    self.options += "':flat', False, "
    self.options += "':ite_extra_rules', True, "
    self.options += "':max_memory', 10737418240, "
    self.options += "':pull_cheap_ite', True, "
    self.options += "':push_ite_arith', True, "
    self.options += "':som', True, "
    self.options += "':som_blowup', 1000, "
    self.options += "':sort_store', True, "
    self.options += "':sort_sums', True, "
    self.options += "':split_concat_eq', True, "
    self.options += "':blast_select_store', True, "
    self.options += "':expand_select_ite', True"
    self.options += ",':hoist_mul', True"
    self.options += ",':hoist_ite', True"

  def assemble_decl (self):
    ret = ""
    for dd in self.decl:
      if (not ret == ""):
        ret += "\n"
      ret = ret + dd
    return ret

  def assemble_cstr (self):
    ret = ""
    for cc in self.cstr:
      if (not ret == ""):
        ret += ", "
      ret = ret + cc
    return ret

  def print_decl_debug (self):
    variables = self.assemble_decl ()
    print ('Declared variables: {}'.format (variables))

  def print_cstr_debug (self):
    constraints = self.assemble_cstr ()
    print ('Formulation : {}'.format (constraints))

  def add_cstr (self, new_cstr):
    self.cstr.append (new_cstr)

  def add_var (self, new_decl):
    self.decl.append (new_decl)

  def write_chunk (self, ff, chunk, chunk_id):
    cmnt = '## Chunk No. {} \n'.format (chunk_id)
    ff.write (cmnt)
    cmd = 'term = simplify (And ({}), {})\n'.format (chunk, self.options)
    ff.write (cmd)
    cmd = 'opt.add (term)\n'
    ff.write (cmd)
    ff.write ('\n')

  ## Write the COF to a python file script.
  def write_formulation (self, glob_obj_ub, n_fails):
    MAX_CHUNK = 150
    variables = self.assemble_decl ()
    constraints = self.assemble_cstr ()
    ff = open (self.modelfile, 'w')
    ff.write ('from z3 import *\n')
    topts = ''
    topts += "':arith.min',True"
    topts += ","
    topts += "':arith.nl.rounds',1048576"
    topts += ","
    topts += "':arith.nl.delay',1000"
    topts += ","
    topts += "':qi.quick_checker',2"
    topts += ","
    topts += "':arith.nl.gr_q',50"
    topts += ","
    topts += "':algebraic_number_evaluator', False, "
    topts += "':arith_ineq_lhs', False, " 
    topts += "':eq2ineq', False, " 
    topts += "':expand_nested_stores', True, "
    topts += "':gcd_rounding', False, "
    topts += "':ignore_patterns_on_ground_qbody', True, "
    topts += "':flat', False, "
    topts += "':ite_extra_rules', True, "
    topts += "':pull_cheap_ite', True, "
    topts += "':push_ite_arith', True, "
    topts += "':som', True, "
    topts += "':som_blowup', 1000, "
    topts += "':sort_store', True, "
    topts += "':sort_sums', True, "
    topts += "':split_concat_eq', True, "
    topts += "':blast_select_store', True, "
    topts += "':expand_select_ite', True"
    topts += ",':hoist_mul', True"
    topts += ",':hoist_ite', True"
    ff.write ("opt = Then('simplify',With('ufnia',{})).solver ()\n".format (topts))
    ff.write ('\n')
    ff.write (variables)
    ff.write ('\n')
    ff.write ('## Formulation Objectives\n')
    if (glob_obj_ub > 0):
      base_scale = n_fails
      left_scale = 1
      right_scale = 1
      if (DIMAGE_FAST_SOL):
        left_scale = base_scale + 2
        right_scale = base_scale + 1
        if (option_dft_conv):
          left_scale = 5
          right_scale = 4
      iter_g_obj_cstr = '{} * G_prog < {} * {}'.format (left_scale, right_scale, glob_obj_ub)
      if (n_fails > 0):
        self.cstr = self.cstr[:-1]
      self.cstr.append (iter_g_obj_cstr)
    ff.write ('\n')
    chunk = ""
    count = 0
    chunk_id = 1
    cache = {}
    for cc in self.cstr:
      if (cc in cache):
        continue
      cache[cc] = 1
      if (count > 0):
        chunk += ", "
      count += 1
      count += cc.count (',') 
      chunk = chunk + cc
      if (count >= MAX_CHUNK):
        self.write_chunk (ff, chunk, chunk_id)
        count = 0
        chunk = ""
        chunk_id += 1
    # Write last chunk
    self.write_chunk (ff, chunk, chunk_id)   
    # Script epilogue
    ff.write ('mod_res = opt.check ()\n')
    ff.write ('if (mod_res == unsat):\n')
    ff.write ('  print("unsat")\n')
    ff.write ('elif (mod_res == unknown):\n')
    ff.write ('  print("unknown [Reason: {}]".format (opt.reason_unknown ()))\n')
    ff.write ('elif (mod_res == sat):\n')
    ff.write ('  sol = opt.model ()\n')
    ff.write ('  for vv in sol:\n')
    ff.write ('    print(vv, sol[vv])\n')
    ff.write ('else:\n')
    ff.write ('  print("Unexpected model state: {}.".format (mod_res))\n')
    ff.close ()



class Dist:
  def __init__(self, pset, stmts, cgstmts, cfilename):
    self.PP = pset
    self.stmts = stmts
    self.CG = cgstmts
    self.cfilename = cfilename
    self.ff = None
    self.arrays = None
    self.producers = {}
    self.last_writer = {}
  
  def writeln (self, line, comment = ''):
    self.ff.write (line + ' ## ' + comment + "\n")

  def empty_line (self):
    self.writeln ('')

  def indent (self):
    self.ff.write ('  ')

  def print_processor_geometry (self):
    sizes = self.PP.get_sizes ()
    for pp in sizes:
      macro_name = self.PP.get_dim_macro_name (pp)
      self.writeln ('#define {} ({})'.format (macro_name, sizes[pp]))

  # Dist.collect_arrays
  def collect_arrays (self):
    ret = {}
    for ss in self.stmts:
      stm = self.stmts[ss]
      ret = stm.collect_arrays (ret)
    return ret

  def collect_last_op_writer (self):
    ret = {}
    for aa in self.arrays:
      ref = self.arrays[aa]
      last = None
      for ss in self.CG:
        if (ss.writes_to (ref)):
          last = ss
      if (last != None):
        ret[aa] = last
    self.last_writer = ret
    for ss in self.CG:
      ss.set_last_writer_map (self.last_writer)
    return ret

    
  def collect_communicators (self):
    comms = {}
    for sid in self.stmts:
      ss = self.stmts[sid]
      comms = ss.collect_communicators (comms)
    return comms


  def declare_communicators (self, comms):
    self.writeln ('// Communicator used in program, one per {array,statement}')
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.declare_communicators (self.ff)

  def declare_used_ispace_dimensions (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_udim_declarations (self.ff)

  def declare_amap_declarations (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      ref.generate_ref_amap_declarations (self.ff)

  def declare_imap_declarations (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_stmt_imap_declarations (self.ff)

  def generate_communicators (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_communicators (self.ff)
    
  ## Dist.generate_operators ():
  ## Generate baseline operators.
  def generate_operators (self):
    #for sid in self.stmts:
    mrap = {} # mrap is "Most Recent Array Producer"
    for ss in self.CG:
      self.empty_line ()
      new_arr = ss.generate_operator (self.ff, self.PP, self.producers, mrap)
      if (new_arr != None and not new_arr in self.producers):
        self.producers[new_arr] = ss
    self.empty_line ()

  ## Dist.generate_single_node_reference_dag ():
  ## Generate single node reference operators to perform a final check.
  def generate_single_node_reference_dag (self):
    self.writeln ('void reference ()')
    self.writeln ('{')
    mrap = {} # mrap is "Most Recent Array Producer"
    decl = '  int {} = 0;'.format (DIMAGE_BLOCK_COUNT)
    self.writeln (decl)
    for ii in range(10):
      it = '  int t{} = 0;'.format (ii)
      self.writeln (it)
      it = '  int i{};'.format (ii)
      self.writeln (it)
    for ss in self.CG:
      self.empty_line ()
      if (not ss.is_data_generator ()):
        self.writeln ('  {')
      new_arr = ss.generate_single_node_operator (self.ff, self.PP, self.producers, mrap)
      if (new_arr != None and not new_arr in self.producers):
        self.producers[new_arr] = ss
      if (not ss.is_data_generator ()):
        self.writeln ('  }')
    self.empty_line ()
    self.writeln ('  // check call goes here')
    self.empty_line ()
    for ss in self.CG:
      if (ss.is_data_generator ()):
        arr_name = 'sna_' + ss.accs[0].get_name ()
        free_call = '  free ({});'.format (arr_name)
        self.writeln (free_call)
    self.writeln ('}\n')


  def declare_arrays (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_name ()
      line = '{} * {};'.format (DIMAGE_DT, varname)
      self.writeln (line)

  def declare_tile_maps (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_tile_map_name ()
      line = '{} * {};'.format (DIMAGE_INT, varname)
      self.writeln (line)


  def codegen (self, sys_argv, avg_call_solv, iters, num_fails, solset):
    self.arrays = self.collect_arrays ()
    dist.collect_last_op_writer ()
    self.ff = open (self.cfilename, "w")
    self.writeln ('// Script {} invoked with : {}'.format (DIMAGE_PY_SCRIPT, sys_argv))
    self.writeln ('// Average solver call: {}; Iterations: {}; Fails = {}\n'.format (avg_call_solv, iters, num_fails))
    self.writeln ('// K_RATIO of communication-to-computation: {};\n'.format (MEM2COMP_RATIO))
    self.writeln ('// Optimal value found (G_prog): {}\n'.format (solset['G_prog']))
    self.writeln ('#include "dimage-rt.h"')
    self.empty_line ()
    self.writeln ('#ifndef DIMAGE_TILE_HEADER_SIZE')
    self.writeln ('#define DIMAGE_TILE_HEADER_SIZE {}'.format (DIMAGE_TILE_HEADER_SIZE))
    self.writeln ('#endif')
    self.empty_line ()
    self.writeln ('int {};'.format (DIMAGE_PROC_RANK))
    self.writeln ('int {}[DIMAGE_MAX_GRID_DIMS];'.format (DIMAGE_PROC_COORDS))
    self.PP.declare_processor_coordinate_variables (self.ff)
    comms = self.collect_communicators ()
    self.empty_line ()
    self.declare_communicators (comms)
    self.empty_line ()
    self.declare_timers ()
    self.empty_line ()
    self.writeln ("// Processor-space grid")
    self.PP.generate_processor_space_declarations (self.ff)
    self.empty_line ()
    self.writeln ("// Iteration-space to processor-space mappings")
    self.declare_imap_declarations ()
    self.empty_line ()
    self.writeln ("// Data-space to processor-space mappings")
    self.declare_amap_declarations ()
    self.empty_line ()
    self.writeln ("// Iteration-space to data-space mappings")
    self.writeln ("// (similar to poly. access functions).")
    self.declare_used_ispace_dimensions ()
    self.print_processor_geometry ()
    self.empty_line ()
    self.writeln ("// Declare arrays as global variables")
    self.declare_arrays ()
    self.empty_line ()
    self.writeln ("// Declare arrays for tile-maps (dictionaries) as global variables")
    self.declare_tile_maps ()
    self.empty_line ()
    self.generate_operators ()
    if (option_check):
      self.generate_single_node_reference_dag ()
    self.generate_main (option_check)
    self.ff.close ()

  def insert_operator_calls (self, option_check):
    self.indent ()
    self.writeln ('// Computing baseline')
    if (option_check):
      self.indent ()
      self.writeln ('reference ();')
      self.empty_line ()
    self.indent ()
    self.writeln ('// Operator calls')
    for ss in self.CG:
      self.indent ()
      self.writeln ('log_msg ("Calling operator {}");'.format (ss.get_name()))
      self.indent ()
      ss.insert_operator_call (self.ff)

  def deallocate_arrays (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_name ()
      line = 'free ({});'.format (varname)
      self.indent ()
      self.writeln (line)

  def deallocate_tile_maps (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_tile_map_name ()
      line = 'free ({});'.format (varname)
      self.indent ()
      self.writeln (line)

  def get_total_computation_timer (self):
    return 'timer_total_computation'

  def get_total_communication_timer (self):
    return 'timer_total_communication'

  def get_generator_computation_timer (self):
    return 'timer_generator_computation'

  def get_generator_communication_timer (self):
    return 'timer_generator_communication'

  def get_operator_computation_timer (self):
    return 'timer_operator_computation'

  def get_operator_communication_timer (self):
    return 'timer_operator_communication'

  def get_full_operator_timer (self):
    return 'timer_operator_full'

  def declare_timers (self):
    self.writeln ('// Declaring timers ')
    self.writeln ('double timer_start;\n')
    self.writeln ('double {} = 0.0;\n'.format (self.get_total_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_total_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_generator_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_generator_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_operator_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_operator_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_full_operator_timer ()))
    for ss in self.CG:
      ss.declare_timer (self.ff)
    self.empty_line ()


  def reduce_timer (self, src_timer, dst_timer, use_sum):
    self.indent ()
    red_op = 'MPI_MAX'
    if (use_sum):
      red_op = 'MPI_SUM'
    self.writeln ('double {};'.format (dst_timer))
    self.indent ()
    self.writeln ('MPI_Reduce (&{}, &{}, 1, MPI_DOUBLE, {}, 0, MPI_COMM_WORLD);'.format (src_timer, dst_timer, red_op))

  def accumulate_timers (self):
    self.empty_line ()
    self.indent ()
    self.writeln ('// Aggregating timers ')
    total_comp_timer = self.get_total_computation_timer ()
    total_comm_timer = self.get_total_communication_timer ()
    generator_comp_timer = self.get_generator_computation_timer ()
    generator_comm_timer = self.get_generator_communication_timer ()
    operator_comp_timer = self.get_operator_computation_timer ()
    operator_comm_timer = self.get_operator_communication_timer ()
    operator_full_timer = self.get_full_operator_timer ()
    for ss in self.CG:
      self.indent ()
      line = '{} += {};'.format (total_comp_timer, ss.get_local_computation_timer ())
      self.writeln (line)
      self.indent ()
      line = '{} += {};'.format (total_comm_timer, ss.get_local_communication_timer ())
      self.writeln (line)
      if (ss.is_data_generator ()):
        self.indent ()
        line = '{} += {};'.format (generator_comp_timer, ss.get_local_computation_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} += {};'.format (generator_comm_timer, ss.get_local_communication_timer ())
        self.writeln (line)
      elif (not ss.is_data_sink ()):
        self.indent ()
        line = '{} += {};'.format (operator_comp_timer, ss.get_local_computation_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} += {};'.format (operator_comm_timer, ss.get_local_communication_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} = {} + {};'.format (operator_full_timer, operator_comm_timer, operator_comp_timer)
        self.writeln (line)
    if (option_include_all):
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Local computation time (sec): %.6lf\\n", {});'.format (total_comp_timer))
      self.indent ()
      self.writeln ('printf ("Local communication time (sec): %.6lf\\n", {});'.format (total_comm_timer))
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Generator-only (local) computation time (sec): %.6lf\\n", {});'.format (generator_comp_timer))
      self.indent ()
      self.writeln ('printf ("Generator-only (local) communication time (sec): %.6lf\\n", {});'.format (generator_comm_timer))
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Operator-only (local) computation time (sec): %.6lf\\n", {});'.format (operator_comp_timer))
      self.indent ()
      self.writeln ('printf ("Operator-only (local) communication time (sec): %.6lf\\n", {});'.format (operator_comm_timer))
      self.indent ()
      self.writeln ('printf ("Operator-only (local) total time (sec): %.6lf\\n", {});'.format (operator_full_timer))
    ## Collect max timers.
    timer_var = 'timer_comp_max'
    self.reduce_timer (operator_comp_timer, timer_var, False)
    timer_var = 'timer_comm_max'
    self.reduce_timer (operator_comm_timer, timer_var, False)
    timer_var = 'timer_total_max'
    self.reduce_timer (operator_full_timer, timer_var, False)
    ## Collect sum timers.
    timer_var = 'timer_comp_sum'
    self.reduce_timer (operator_comp_timer, timer_var, True)
    timer_var = 'timer_comm_sum'
    self.reduce_timer (operator_comm_timer, timer_var, True)
    timer_var = 'timer_total_sum'
    self.reduce_timer (operator_full_timer, timer_var, True)
    self.indent ()
    self.writeln ('if (dimage_rank == 0) {')
    self.indent ()   
    timer_var = 'timer_comp_max'
    self.writeln ('  printf ("Max. compute-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comm_max'
    self.writeln ('  printf ("Max. communication-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_total_max'
    self.writeln ('  printf ("Max. total-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comp_sum'
    self.writeln ('  printf ("Avg. compute-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comm_sum'
    self.writeln ('  printf ("Avg. communication-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_total_sum'
    self.writeln ('  printf ("Avg. total-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()
    self.writeln ('}')
    

  def generate_main (self, option_check):
    self.writeln ('int main(int argc, char** argv) {')
    self.indent ()
    self.writeln ('MPI_Init(NULL, NULL);')
    self.empty_line ()
    self.indent ()
    self.writeln ('int dimage_cw;')
    self.indent ()
    self.writeln ('MPI_Comm_size (MPI_COMM_WORLD, &dimage_cw);')
    self.indent ()
    self.writeln ('MPI_Comm_rank(MPI_COMM_WORLD, &{});'.format (DIMAGE_PROC_RANK))
    self.PP.init_processor_coordinates (self.ff)
    self.indent ()
    n_proc_dim = self.PP.get_num_dim ()
    proc_coord_list = self.PP.get_processor_coordinate_str_list ()
    self.writeln ('init_log_file_{}D_with_rank ({}, {});'.format (n_proc_dim, proc_coord_list, DIMAGE_PROC_RANK))
    self.indent ()
    self.writeln ('int comm_color;')
    self.indent ()
    self.writeln ('int comm_vec[{}];'.format (self.PP.get_num_dim ()))
    self.empty_line ()
    self.generate_communicators ()
    self.empty_line ()
    self.insert_operator_calls (option_check)
    self.indent ()
    self.empty_line ()
    self.deallocate_arrays ()
    self.empty_line ()
    self.deallocate_tile_maps ()
    self.empty_line ()
    self.indent ()
    self.accumulate_timers ()
    self.empty_line ()
    self.indent ()
    self.writeln ('MPI_Finalize ();')
    self.indent ()
    self.writeln ('return 0;')
    self.writeln ('}');

  ## Generate a Makefile specific to the input *.rel file.
  def gen_makefile (self):
    mf = open('Makefile','w')
    mf.write ('MPICC=mpicc\n')
    mf.write ('MPIRUN=mpirun\n')
    mf.write ('MPIOPTS=-np {} --use-hwthread-cpus --oversubscribe\n'.format (self.PP.get_max_procs ()))
    procs=self.PP.get_max_procs ()
    mf.write ('OSCRUN=srun\n')
    mf.write ('OSCOPTS=--nodes={} --ntasks={} --ntasks-per-node=1 --cpus-per-task=28\n'.format (procs, procs))
    mf.write ('DIMAGERT=dimage-rt.c\n')
    debug_opts = ' -D DIMAGE_LOG -D DIMAGE_DEBUG -D USE_INIT_DIAGONAL -D INIT_MAT'
    defs_bench = ' -D DIMAGE_TILE_HEADER_SIZE={} -D DIMAGE_KERNEL_LOOP  '.format (DIMAGE_TILE_HEADER_SIZE)
    defs_loop = ' -D DIMAGE_TILE_HEADER_SIZE={} -D DIMAGE_KERNEL_LOOP -D INIT_MAT '.format (DIMAGE_TILE_HEADER_SIZE)
    defs_no_loop = ' -D DIMAGE_TILE_HEADER_SIZE={} '.format (DIMAGE_TILE_HEADER_SIZE)
    check_debug_opts = ' -D DIMAGE_LOG -D DIMAGE_DEBUG -D INIT_MAT'
    mf.write ('DEBUGOPTS={}\n'.format (debug_opts))
    mf.write ('CHECK_DEBUG_OPTS={}\n'.format (check_debug_opts))
    mf.write ('SRCS={} code/dimage-rt.c\n'.format (self.cfilename))
    bin_name = re.sub ('\.c','.exe',self.cfilename)
    bin_debug_name = re.sub ('\.c','.debug.exe',self.cfilename)
    bin_debug_loop_name = re.sub ('\.c','.debug-loop.exe',self.cfilename)
    mklflags=' -O3 -qopenmp -fPIC -lmpi -ilp64 -lmpi_ilp64 -mkl=parallel -D DIMAGE_MKL $(MKL_CLUSTER_LIBS) '
    intelflags=' -O3 -qopenmp -fPIC -lmpi -ilp64 -lmpi_ilp64  '
    mf.write ('\n')
    mf.write ('all: dist\n')
    mf.write ('\n')
    mf.write ('dist: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_bench, bin_name))
    mf.write ('\n')
    mf.write ('mkl: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\tmpiicc -I. -I code {} $(SRCS) {} -o {} \n'.format (mklflags, defs_no_loop, bin_name))
    mf.write ('\n')
    mf.write ('intel: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\tmpiicc -I. -I code {} $(SRCS) -o {} \n'.format (intelflags,bin_name))
    mf.write ('\n')
    mf.write ('debug: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(DEBUGOPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_no_loop, bin_debug_name))
    mf.write ('\n')
    mf.write ('debug-loop: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(DEBUGOPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_loop, bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('debug-loop-hard: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(CHECK_DEBUG_OPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_loop, bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('bench:\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_name))
    mf.write ('\n')
    mf.write ('run-osc:\n')
    mf.write ('\t$(OSCRUN) $(OSCOPTS) ./{}\n'.format (bin_name))
    mf.write ('\n')
    mf.write ('check-debug: gendata baseline debug\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_name))
    mf.write ('\n')
    mf.write ('check-debug-loop: gendata baseline debug-loop\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('check-hard-debug-loop: gendata-hard baseline-hard debug-loop-hard\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('baseline:\n')
    for ss in self.CG:
      if (ss.is_data_generator () or ss.is_data_sink ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c $(DEBUGOPTS) {} -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('baseline-hard:\n')
    for ss in self.CG:
      if (ss.is_data_generator () or ss.is_data_sink ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c $(CHECK_DEBUG_OPTS) {} -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('gendata:\n')
    for ss in self.CG:
      if (not ss.is_data_generator ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c {} $(DEBUGOPTS) -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('gendata-hard:\n')
    for ss in self.CG:
      if (not ss.is_data_generator ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c {} $(CHECK_DEBUG_OPTS) -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('clean:\n')
    mf.write ('\trm -f {} *.data phases_* *.mat\n'.format (bin_name))
    mf.close ()

    

class Reference:
  def __init__(self, form, PP, NP):
    self.name = ""
    self.np = NP
    self.PP = PP
    self.ndim = 0
    self.cof = form
    self.dims = {}  # Dimension indices hashed by dimension number. Value is iterator name.
    self.sizes = {}
    self.map = {}
    self.data = None
    self.uses_slice = False
    self.last_access = 'ERROR'
    self.is_allgat_out_slice = False
    self.is_allgat_in_slice = False
    self.free_list = ''
    self.lin_groups = []
    self.parent_operator = None

  ## Reference
  def tensor_init_from_file (self, ff):
    line = ff.readline ()
    # Expected format: <tensor_name>:<ordered-access-list>:<extents>
    line = line.strip ()
    parts = line.split (':')
    self.name = parts[0]
    if (self.name.find ('_') >= 0):
      print ('Use of underscore (_) character is not permitted.')
      sys.exit (42)
    # Expect iterator list separated by ';'
    # or fused iterators separated by ','
    dimlist = parts[1].split (RELS_LIST_SEPARATOR)
    dcount = 0
    ## Parse the list of linearized accessors. 
    ## Dimensions separated by '*'. Assume the order in which iterators
    ## appear matters.
    self.lin_groups = []
    for acc_expr in dimlist:
      indices = acc_expr.split ('*')
      group = []
      for iter_name in indices:
        self.dims[dcount] = iter_name
        self.map[dcount] = DIM_UNMAPPED
        group.append (dcount)
        dcount += 1
        self.ndim += 1
      self.lin_groups.append (group[:])
    sizes = parts[2].split (RELS_LIST_SEPARATOR)
    for dd,dsize in enumerate(sizes):
      self.sizes[dd] = dsize

  def set_parent_operator (self, parent):
    self.parent_operator = parent

  ## Retrieve the most recently generated access.
  def get_precollective_buffer_access (self):
    return self.last_access

  ## Reference.set_precollective_buffer_access ():
  ## Store the most recently generated access.
  def set_precollective_buffer_access (self, new_acc):
    self.last_access = new_acc

  def set_is_allgat_out_slice (self, bool_val):
    self.is_allgat_out_slice = bool_val

  def get_is_allgat_out_slice (self):
    return self.is_allgat_out_slice

  def set_is_allgat_in_slice (self, bool_val):
    self.is_allgat_in_slice = bool_val

  def get_is_allgat_in_slice (self):
    return self.is_allgat_in_slice
  

  def get_matrix_filename (self, op_name = ''):
    atop = ''
    if (op_name != ""):
      atop = '_at_{}'.format (op_name)
    fname = '{}{}.mat'.format (self.name, atop)
    return fname

  def estimate_memory_requirements (self):
    vol = 1
    for dsize in self.sizes:
      vol *= int(self.sizes[dsize])
    return vol

  def get_data (self):
    return self.data

  def get_dims (self):
    return self.dims

  ## Reference.
  ## FFT
  def is_viable (self, avg_proc_dim):
    ret = True
    for idx,dsize in enumerate(self.sizes):
      dim_res = int(self.sizes[dsize]) >= int(avg_proc_dim)
      print ('Testing viabily of {}[{}]={} --> {} = {} >= {}'.format (self.name, idx, self.sizes[dsize], dim_res, self.sizes[dsize], avg_proc_dim))
      ret = ret and dim_res
    return ret

  ## Reference.
  ## FFT
  ## Return a product of tensor extents, but only include the extent if the 
  ## corresponding entry in the vector pmode is 1.
  def estimate_local_computational_workload (self, pmode):
    ret = ''
    #print ('Parallel mode to estimate workload = {}'.format (pmode))
    for dd in range(len(pmode)):
      if (pmode[dd] == 1):
        continue
      if (ret != ''):
        ret += ' * '
      ret += str(self.sizes[dd])
    return ret

  ## Reference
  ## Check if last dimension is a batch dimension.
  def get_num_batch_dim (self):
    for dd in self.dims:
      if (self.dims[dd].find ('#') > 0):
        return 1
    return 0

  def get_dim_name (self, adim):
    if (adim >= len(self.dims)):
      sys.exit (42)
    return self.dims[adim]

  def show_data (self):
    if (self.data == None):
      print ("[{}]: No data found".format (self.name))
      return
    print ("[{}] data:".format (self.name))
    N0 = int(self.sizes[0])
    N1 = int(self.sizes[1])
    for ii in range(N0):
      line = ""
      for jj in range(N1):
        line += "{:.6} ".format (self.data[ ii * N1 + jj])
      print (line)
    print ()
  
  def gen_matrix_data (self):
    fname = self.get_matrix_filename ()
    mat = open (fname, 'w')
    if (self.ndim == 2):
      N0 = int(self.sizes[0])
      N1 = int(self.sizes[1])
      self.data = [0] * (N0 * N1)
      for ii in range(N0):
        for jj in range(N1):
          val = ((ii + (abs(self.map[1]) + 1.0) * N1) * 1.0 + (jj + ord(self.name[0])) + abs(self.map[0]) + 1.0) / (N0 * N1 * 1.0)
          mat.write ('{:.6f} '.format (val))
          index = ii * N1 + jj
          self.data[index] = val
        mat.write ('\n')
    if (self.ndim == 1):
      N0 = int(self.sizes[0])
      self.data = [0] * (N0)
      for ii in range(N0):
        val = (ii * 1.0)  / (N0 * 1.0)
        mat.write ('{} '.format (val))
        index = ii
        self.data[index] = val
    if (self.ndim == 3):
      N0 = int(self.sizes[0])
      N1 = int(self.sizes[1])
      N2 = int(self.sizes[2])
      self.data = [0] * (N0 * N1 * N2)
      for ii in range(N0):
        for jj in range(N1):
          for kk in range(N2):
            val = ((((ii + kk) % N1) * N1) * 1.0 + jj * N2 + ((kk + jj) % N1)) / (ii * kk + jj * N2 + 1.0)
            mat.write ('{} '.format (val))
            index = ii * N1 * N2 + jj * N2 + kk
            self.data[index] = val
          mat.write ('\n')
        mat.write ('\n')
    mat.close ()
  
  ## Show the array name and its sizes by printing it to stdout.
  def show_info (self):
    nbatchdim = self.count_batch_dims ()
    print ("  Reference: {} (ndim={},batch_dim={})".format (self.name, self.ndim, nbatchdim))
    for dd in self.dims:
      print ("  --> Dim {}: {} ({})".format (dd, self.dims[dd], self.sizes[dd]))
    print ("  --> Lin.Groups = {}".format (self.lin_groups))

  ## Return the reference as a string. Used for debugging.
  def get_as_str (self):
    ret = ''
    for dd in self.dims:
      if (ret != ''):
        ret += ','
      ret += str(self.dims[dd])
    ret = '{}[{}]'.format (self.name, ret)
    return ret



  def get_pi_map (self):
    return self.map

  ## Reference
  ## Minor tweak to provide more information about not found DS dimension.
  def get_pi_var_by_dim_name (self, dim_name, pdim):
    ret = ''
    for adim in self.dims:
      if (self.dims[adim] == dim_name):
        return self.get_map_varname (adim, pdim)
    return 'ERROR({} not found)'.format(dim_name)

  def is_fully_replicated (self):
    for dd in self.dims:
      if (self.map[dd] >= 0):
        return False
    return True
    
  def is_replicated_at_dim (self, adim):
    if (adim >= len(self.map)):
      print ('ERROR @ is_replicated_at_dim ')
      sys.exit (42)
    return self.map[adim] == -1

  def is_pi_map_dim_equal (self, iter_name, mu_pdim):
    for dd in self.dims:
      if (self.dims[dd] == iter_name):
        if (self.map[dd] == mu_pdim and mu_pdim >= 0):
          print ("\t\t Reference {} at dimension {} matched proc.dim {}".format (self.name, dd, mu_pdim))
          return True
        return False
    return False

  def is_mu_map_dim_strict_subset_of_pi (self, iter_name, mu_pdim):
    for dd in self.dims:
      if (self.dims[dd] == iter_name):
        if (self.map[dd] == -1 and mu_pdim >= 0):
          return True
        return False
    return False

  def show_maps (self):
    print ("\tArray {} mappings".format (self.name))
    for dd in self.map:
      print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))

  # Draw tikz graph for statement using its mapping
  def print_tikz_graph (self, fout, par_x, par_y):
    for dd in self.map:
      pi = self.map[dd]
      nodename='{}_i{}'.format (self.name, dd)
      dimname=self.dims[dd]
      nodelabel = '{\\large\\textbf{ ' + '{}[i{}]'.format (self.name, dd) + '}}'
      if (pi < 0):
        nodelabel = '{\\large\\textbf{' + '{}[i{}]=*'.format (self.name, dd) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node[shape=rectangle,draw=red,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      if (pi >= 0):
        procdim = 'p{}'.format (pi)
        src = '({}.west)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[right] {} {};'.format ('->,line width=1mm,red',src,'{}',tgt)
        fout.write (command + '\n')
    return len(self.map)


  def get_ref_dim (self):
    return len(self.dims)

  def pretty_print_map (self, df):
    df.write ('<')
    for dd in self.map:
      if (dd > 0):
        df.write (', ')
      map_dim = self.map[dd]
      if (map_dim >= 0):
        df.write ('{}'.format (map_dim))
      else:
        df.write ('{}=*'.format (map_dim))
    df.write ('>')

  ## Method to add a generic pre-assembled constraint to the COF object
  ## and to the formulation file.
  def add_constraint (self, mf, cstr, info = ''):
    comment = ''
    if (info != ''): 
      comment = ' ## ' + info
    self.writeln (mf, 'opt.add ({}) {}'.format (cstr, comment))
    self.cof.add_cstr (cstr)

  ## Reference.get_name (): Return the name of the array
  def get_name (self):
    return self.name

  ## Reference.get_tile_map_name (): Return the name of tile-map
  ## variable / dictionary.
  def get_tile_map_name (self, ext_buffer = None):
    if (ext_buffer != None):
      return 'TM_{}'.format (ext_buffer)
    return 'TM_{}'.format (self.name)


  def get_tile_name (self, intermediate = None):
    if (intermediate == None):
      return 'tile_{}'.format (self.name)
    return 'tile_{}'.format (intermediate)

  def get_sna_ref_name (self, intermediate = None):
    if (intermediate == None):
      return 'sna_{}'.format (self.name)
    return 'sna_{}'.format (intermediate)

  def get_sna_reference_filename (self):
    return 'ref_{}'.format (self.name)

  def get_tile_header_size (self, all_tiles):
    proc_geom = self.PP.get_processor_geometry_list_from_map (self.map, all_tiles)
    proc_geom = re.sub (',',' *', proc_geom)
    return '({} * {})'.format (DIMAGE_TILE_HEADER_SIZE, proc_geom)

  ## Reference.get_name_for_check (): Return the name for the reference array
  ## used in calls to check_arrayND().
  def get_name_for_check (self):
    return 'ref_{}'.format (self.name)

  ## Return the number of dimensions of the current array object.
  def get_num_dim (self):
    return len(self.dims)

  def get_iter_name (self, adim):
    return self.dims[adim]

  ## Reference.
  def get_map_varname (self, idim, pdim):
    varname = 'pi_{}_i{}_p{}'.format (self.name, idim, pdim)
    return varname
    
  #def writeln(self, mf, line):
  #  mf.write (line + "\n")
  #
  def writeln (self, mf, line, comment = ''):
    mf.write (line + ' ## ' + comment + "\n")

  def set_lower_bound (self, mf, varname, lb):
    cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_upper_bound (self, mf, varname, ub):
    cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  # Reference
  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    self.set_bounds (mf, varname, lb, ub)

  def is_dimension_mapped (self, adim):
    if (self.map[adim] >= 0):
      return True
    return False

  def get_array_communicator_at_statement (self, stmt_name):
    varname = 'dimage_comm_{}_at_{}'.format (self.name, stmt_name)
    return varname

  def get_dimension_communicator_at_statement (self, adim, stmt_name):
    dim_name = 'dimage_comm_{}_dim{}_at_{}'.format (self.name, adim, stmt_name)
    return dim_name

  def collect_communicators_for_statement (self, stmt_name, comms):
    for dd in self.map:
      if (self.is_dimension_mapped (dd)):
        comm_name = self.get_dimension_communicator_at_statement (dd, stmt_name)
        comms[comm_name] = comm_name
    return comms

  ## Reference
  def declare_variable (self, mf, varname, decl):
    if (decl == None):
      print ("Exiting")
      sys.exit(42)
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname  
    return decl

  ## Reference.
  ## Add a \pi (boolean) variable to the COF object.
  ## Declare pi-mapping variables.
  def declare_map_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Dictionary is None")
      sys.exit (42)
    NP = self.np
    ## Separate cases of linearized and non-linearized tensors. Check via self.lin_groups.
    if (len(self.dims) == len(self.lin_groups)):
      for dd in self.dims:
        for pp in range(NP):
        #print ("Dim {}: {}".format (dd, self.dims[dd]))
          varname = self.get_map_varname (dd, pp)
          #print ("Varname received: {}".format (varname))
          decl = self.declare_variable (mf, varname, decl)
          self.set_bounds_boolean (mf, varname)
    else: 
      for dd in range(len(self.lin_groups)):
        for pp in range(NP):
          varname = self.get_map_varname (dd, pp)
          decl = self.declare_variable (mf, varname, decl)
          self.set_bounds_boolean (mf, varname)
    return decl

  ## Reference.get_sum_pi_var_along_dim ():
  ## Sum pi vars along all the processors p of iter space dimension i
  ## or sum all the pi vars along the same processor-space p dimension.
  def get_sum_pi_var_along_dim (self, idim, pdim):
    varname = ''
    if (idim < 0 and pdim < 0):
      print ("[ERROR] Both in [{}]. Both idim and pdim are -1".format ('get_sum_pi_var_along_dim'))
      sys.exit (42)
    if (idim == -1):
      varname = 'sum_pi_{}_iX_p{}'.format (self.name, pdim)
    else:
      varname = 'sum_pi_{}_i{}_pX'.format (self.name, idim)
    return varname

  ## Reference.
  ## Sum all the pi variables along an iteration-space dimension or
  ## along a processor-space dimension.
  def set_sum_bound_along_dim (self, mf, mode, dim, ub, decl):
    nn = min(self.ndim,len(self.lin_groups)) ## Min of number of dimensions and linearized groups.
    #nn = self.ndim
    if (mode == PER_DIM):
      nn = self.np
    if (mode == PER_DIM):
      self.writeln (mf, '## per dim: np = {}'.format (self.np))
    else:
      self.writeln (mf, '## per proc: np = {}'.format (self.ndim))
    cstr = ""
    # By default, assume we are summing along all the iteration-space 
    # variables for a fixed processor.
    pi_sum_var = self.get_sum_pi_var_along_dim (-1, dim)
    if (mode == PER_DIM):
      pi_sum_var = self.get_sum_pi_var_along_dim (dim, -1)
    decl = self.declare_variable (mf, pi_sum_var, decl)
    for kk in range(nn):
      if (not cstr == ""):
        cstr += " + "
      varname = ""
      if (mode == PER_DIM):
        varname = self.get_map_varname (dim, kk)
      if (mode == PER_PROC):
        varname = self.get_map_varname (kk, dim)
      cstr += varname
    cstr = '{} == {}'.format (pi_sum_var, cstr)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    cstr = '{} >= 0, {} <= {}'.format (pi_sum_var, pi_sum_var, ub)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd, ' set_sum_bound_along_dim ' )
    self.cof.add_cstr (cstr)
    return decl

  def get_max_linear_group_size (self):
    ret = 0
    for group in self.lin_groups:
      ret = max(ret,len(group))
    return ret


  ## Reference.
  ## FFT.
  ## Added for DiMage-FFT.
  ## Sum all the pi variables along an iteration-space dimension or
  ## along a processor-space dimension.
  def set_linearized_sum_bound_along_dim (self, mf, mode, dim, ub, decl):
    ## In PER_PROC mode, initialize nn with ndim of reference.
    nn = min(self.ndim,len(self.lin_groups))
    if (mode == PER_DIM):
      nn = self.np
    if (mode == PER_DIM):
      self.writeln (mf, '## per dim: np = {}'.format (self.np))
    else:
      self.writeln (mf, '## per proc: np = {}'.format (self.ndim))
    sum_cstr = ''
    # By default, assume we are summing along all the iteration-space 
    # variables for a fixed processor.
    pi_sum_var = self.get_sum_pi_var_along_dim (-1, dim)
    if (mode == PER_DIM):
      pi_sum_var = self.get_sum_pi_var_along_dim (dim, -1)
    decl = self.declare_variable (mf, pi_sum_var, decl)
    for lg in self.lin_groups:
      prod_expr = ''
      for idx in lg:
        if (prod_expr != ''):
          prod_expr += ' + '
        varname = ""
        if (mode == PER_DIM):
          varname = self.get_map_varname (dim, idx)
        if (mode == PER_PROC):
          varname = self.get_map_varname (idx, dim)
        prod_expr += varname
      if (sum_cstr != ''):
        sum_cstr += ' + '
      sum_cstr += prod_expr
    ## Compute upper bound for sum of pi along a fixed proc. dimension
    if (mode == PER_PROC):
      ub = 1 #len(self.dims) - 1
    sum_cstr = '{} == {}'.format (pi_sum_var, sum_cstr)
    cmd = "opt.add ({})".format (sum_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (sum_cstr)
    sum_cstr = '{} >= 0, {} <= {}'.format (pi_sum_var, pi_sum_var, ub)  ## Set in previous condition.
    cmd = "opt.add ({})".format (sum_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (sum_cstr)
    return decl

  ## Reference
  ## Return the product of H_p{k} variables corresponding to the minimum
  ## number of processor dimensions in use.
  def get_active_processors (self, dd, pmode):
    term_list = []
    distributed = len(pmode) - vector_mapped_dims (pmode)
    num_batch_dim = self.count_batch_dims ()
    non_batch_dims = self.PP.get_num_dim () - num_batch_dim
    for pp in range(min(distributed,non_batch_dims)):
      proc_var = self.PP.get_proc_dim_symbol (pp)
      term_list.append (proc_var)
    return build_product_from_list (term_list)


  ## Reference
  ## Only an FFT operator should call this function.
  def estimate_communication_size (self, pmode):
    if (not self.parent_operator.is_fft ()):
      print ('Parent operator using tensor {} is not an FFT operator.'.format (self.name, self.parent_operator.get_name ()))
    if (len(pmode) != len(self.dims) - self.parent_operator.nbatchdim):
      print ('Tuple size of parallel mode vector should be of same length as tensor ({} vs {}). Aborting ...'.format (len(mode),len(self.dims)))
      sys.exit (42)
    cstr = self.get_active_processors (-1, pmode)
    return '(' + cstr + ')'

  ## Reference.
  def set_dim_sum_bounds (self, mf, decl):
    for dd in range(min(self.ndim,len(self.lin_groups))):  ## Modification for FFT: min_of(self.dim and self.lin_groups)
      decl = self.set_sum_bound_along_dim (mf, PER_DIM, dd, 1, decl)
    return decl

  ## Reference.
  def set_proc_sum_bounds (self, mf, decl):
    ub = 1 
    for dd in range(self.np):
      decl = self.set_sum_bound_along_dim (mf, PER_PROC, dd, ub, decl) 
    return decl

  ## Reference
  ## Added for DiMage-FFT.
  def set_linearized_proc_sum_bounds (self, mf, decl):
    ub = 1 
    for dd in range(self.np):
      decl = self.set_linearized_sum_bound_along_dim (mf, PER_PROC, dd, ub, decl)   # ub was 1
    return decl

  ## Reference - FFT.
  def get_lincost_var (self, adim, mdim, pp):
    varname = 'LinCost_{}_s{}_t{}_p{}'.format (self.name, adim, mdim, pp)
    return varname

  def get_pi_product (self, adim, pp):
    ret = ''
    #print ('Reference={}, num.groups={}, group={}'.format (self.name, len(self.lin_groups), adim))
    group = self.lin_groups[adim]
    for dd in group:
      if (ret != ''):
        ret += ' * '
      ret += self.get_map_varname (dd, pp)
    return ret
    
  def linearizer_cost_var (self, other):
    return 'LinCost_{}_to_{}'.format (self.name, other.get_name ())

  ## Reference
  def get_effective_processor_count_in_batch (self, fft_stmt):
    num_procs = self.PP.get_max_procs ()
    if (fft_stmt != None and fft_stmt.nbatchdim > 0):
      num_procs = self.PP.get_num_procs_per_batch_dim ()
    return str(num_procs)

  ## Reference
  ## Linearizer constraints for converting N-dimensional tensors to 2D matrices.
  def set_linearizer_communication_constraint (self, mf, decl, nlref, oper_dims, fft_stmt):
    lc_varlist = []
    lc_varlist_matrix = []
    n_dim_tensor = self.ndim
    n_dim_matrix = len(nlref.lin_groups)
    dim_type_map = {}
    for idim in oper_dims:
      iter_name = oper_dims[idim]
      adim_src = self.get_dim_if_used_in_linear_group (iter_name)
      adim_tgt = nlref.get_dim_if_used_in_linear_group (iter_name)
      if (adim_src < 0 or adim_tgt < 0):
        continue
      for pdim in range(self.PP.get_num_dim ()):
        varname = self.get_lincost_var (adim_src, adim_tgt, pdim)
        dim_penalty = 1
        dim_type_map[varname] = dim_penalty
        decl = self.declare_variable (mf, varname, decl)
        lc_varlist.append (varname)
    for lcvar in lc_varlist:
      cstr = '{} >= 0'.format (lcvar)
      self.add_constraint (mf, cstr)
      cstr = '{} <= ((1))'.format (lcvar)  
      self.add_constraint (mf, cstr)
    sum_str = ''
    match_vars = {}
    fact_list = []
    amp_list = []
    for pp in range(self.PP.get_num_dim ()):
      match_vars[pp] = []
      pp_list = []
      dim_penalty = 1
      if (pp == self.PP.get_num_dim () - 1):
        dim_penalty = 2
      for idim in oper_dims:
        iter_name = oper_dims[idim]
        adim_src = self.get_dim_if_used_in_linear_group (iter_name)
        adim_tgt = nlref.get_dim_if_used_in_linear_group (iter_name)
        if (adim_src < 0 or adim_tgt < 0):
          continue
        pi_prod = self.get_pi_product (adim_src, pp)
        pi_cons = nlref.get_map_varname (adim_tgt, pp)
        lcvar = self.get_lincost_var (adim_src, adim_tgt, pp)
        dim_penalty = dim_type_map[lcvar]
        amp_term = '(1 - ({}))'.format (lcvar)
        amp_term = '{} * (1 - ({}))'.format (dim_penalty, lcvar)
        amp_list.append (amp_term)
        cstr = '{} == (1 - ({}) + ({})*({}))'.format(lcvar, pi_prod, pi_cons, pi_prod) ##  \prod [ (1 - pi_prod) + H * (pi_prod)]
        self.add_constraint (mf, cstr)
        pp_list.append (lcvar)
      proc_var = self.PP.get_proc_dim_symbol (pp)
      prod_expr = build_product_from_list (pp_list)
      cstr = '(1 - {}) + {} * ({})'.format (prod_expr, proc_var, prod_expr)
      fact_list.append (cstr)
    prod_expr = build_product_from_list (fact_list)
    amplifier_factor = build_sum_from_list (amp_list)

    data_slice_factor = self.get_fft_min_tensor_volume_expr (fft_stmt)
    num_procs = self.get_effective_processor_count_in_batch (fft_stmt)

    LC_UB = 0
    for lcvar in dim_type_map:
      LC_UB += dim_type_map[lcvar]
    lcvar_main = self.linearizer_cost_var (nlref)
    decl = self.declare_variable (mf, lcvar_main, decl)
    tensor_vol = self.get_full_tensor_volume ()
    data_slice_factor = self.parent_operator.get_rotation_comm_unit ()
    # NOTE: Amplifier considers number of possible mismatches along grid dimensions.
    cstr = '{} == ({} * {} * ({})) / ({})'.format (lcvar_main, tensor_vol, MEM2COMP_RATIO, amplifier_factor, prod_expr)  # Amplified linearizer cost.
    self.add_constraint (mf, cstr, 'linearizer comm. cost on pi-vars')
    cstr = '{} >= {}'.format (lcvar_main, 0) #data_slice_factor)
    self.add_constraint (mf, cstr, " lincost lower bound ")
    cstr = '{} <= {} * {} * {}'.format (lcvar_main, MEM2COMP_RATIO, tensor_vol, LC_UB)  
    self.add_constraint (mf, cstr, " lincost upper bound ")  
    return decl

  ## FFT
  ## Upper-bound pi-mapping variables with the successor-sum variable.
  ## Add all constraints to cstr_list.
  ## Argument @vec is a vector with as many components as FFT dimensions, 
  ## excluding the batch dimension.
  ## If component k of @vec is 0, then we add the constraint: pi^{T,k}_{p} <= 1 - succ_sum_{vec}
  def bound_pi_mapping_by_parmode_vector (self, vec, src_succ_sum, tgt_succ_sum, cstr_map, other):
    for adim in range(len(self.dims) - self.count_batch_dims ()):
      # Condition checks for 1 because a locally mapped dimension cannot be used for distribution.
      sum_pi_var = self.get_sum_pi_var_along_dim (adim, -1)
      e_i = vec[adim]
      if (e_i == 0):
        cstr = '{} * (1 - ({}))'.format (src_succ_sum, tgt_succ_sum)
        if (not sum_pi_var in cstr_map):
          cstr_map[sum_pi_var] = []
        cstr_map[sum_pi_var].append (cstr)
    return cstr_map

      
  def link_dimensions (self, mf, pp, dd, dim, muvar):
    for dd in range(self.ndim):
      if (self.dims[dd] == dim):
        pivar = self.get_map_varname (dd, pp)
        cstr = '{} >= {}'.format (muvar, pivar)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  # Return variable for computing data slices.
  def get_block_size_var (self, dim):
    varname = 'DS_{}_{}'.format (self.name, dim)
    return varname

  def get_rho_varname (self):
    return 'rho_{}'.format (self.name)

  def get_rho_dim_varname (self, adim):
    return 'rho_{}_i{}'.format (self.name, adim)

  # Declare all rho variables for current array.
  def declare_replication_variables (self, mf, decl, LT):
    if (self.get_name () in LT):
      print ("Found {}, ...".format (self.get_name ()))
      return decl
    rho_var = self.get_rho_varname ()
    decl = self.declare_variable (mf, rho_var, decl)
    for dd in self.dims:
      rho_var = self.get_rho_dim_varname (dd)
      decl = self.declare_variable (mf, rho_var, decl)
    return decl

  def bound_replication_variables (self, mf):
    rho_var = self.get_rho_varname ()
    self.set_bounds_boolean (mf, rho_var)
    for dd in self.dims:
      rho_var = self.get_rho_dim_varname (dd)
      self.set_bounds_boolean (mf, rho_var)

  ## Link the rho variable of an array to all of its rho_<array>_dim variables.
  def link_rho_variables (self, mf):
    main_rho_var = self.get_rho_varname ()
    USE_INEQ = False
    if (USE_INEQ):
      for dd in self.dims:
        rho_dim_var = self.get_rho_dim_varname (dd)
        cstr = '{} <= {}'.format (main_rho_var, rho_dim_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)
    else:
      cstr = ''
      for dd in self.dims:
        if (cstr != ''):
          cstr += ' * '
        rho_dim_var = self.get_rho_dim_varname (dd)
        cstr += rho_dim_var
      cstr = '{} == {}'.format (main_rho_var, cstr)
      cmd = 'opt.add ({}) ## link_rho_variables'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)

  ## Reference.define_rho_expression ():
  ## The below function is buggy. It allows the rho variable to be 1,
  ## when any of the placement variables is 1.
  def define_rho_expression (self, mf, adim, rho_dim_var):
    cstr = ''
    sum_pi_var = self.get_sum_pi_var_along_dim (adim, -1)
    ## NOTE: An array dimension is replicated if it's not mapped.
    ## To truly represent replication we define each rho_Ref_dim variable as
    ## rho_Ref_dim == 1 - sum of same rhos along the same dim across all p-dimensions.
    expr = '{} == 1 - {}'.format (rho_dim_var, sum_pi_var)
    cstr = 'opt.add ({})'.format (expr)
    self.writeln (mf, cstr + ' ## Replication <--> \sum pi')
    self.cof.add_cstr (expr)

  ## Reference.define_rho_expression_new ():
  def define_rho_expression_new (self, mf, adim, rho_dim_var):
    cstr = ''
    for pp in range(self.np):
      pi_var = self.get_map_varname (adim, pp)
      cstr = '{} >= (1 - {})'.format (rho_dim_var, pi_var)
      cmd_cstr = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd_cstr)
      self.cof.add_cstr (cstr)

  ## Reference.link_replication_to_placement () : 
  ## Create a constraint where the replication variable, \rho^{A}_{l}, 
  ## upper-bounds the placement variables. When the \rho variable is set
  ## to 1, all of the placement variables, the \pi^{S}_{l,p}, will become 0,
  ## meaning that the array A will not be replicated along any processor-space
  ## dimension p. Further, if an array A is replicated along all space dimensions,
  ## then the array is effectively replicated.
  def link_replication_to_placement (self, mf):
    self.writeln (mf, '## Replication expression of array {}: prod (1-pi^F_[k,p])'.format (self.name))
    for dd in range(min(len(self.dims),len(self.lin_groups))):  ## Modification for FFT: min of len(self.dims) and len(self.lin_groups)
      ## NOTE: In some cases we could prefer un-replicated (distributed arrays). 
      ## Replication often comes with communication cost in the form of all-reduce.
      rho_dim_var = self.get_rho_dim_varname (dd)
      self.define_rho_expression (mf, dd, rho_dim_var)


  ## Return the array dimension size, as an integer, given the associated
  ## iterator name used to access it in the rels input file.
  ## Modification for FFT-DiMage: Added search for iterator name used
  ## in self.lin_groups.
  def get_array_extent_by_dim_name (self, dim_name):
    for dd in range(min(self.ndim,len(self.sizes))):
      if (self.dims[dd] == dim_name):
        return int(self.sizes[dd])
    ## If we didn't find a dimension name with the exact name, then search
    ## in self.lin_groups for a dimension that uses the given iterator name.
    for gid, group in enumerate(self.lin_groups):
      for gg in group:
        if (self.dims[gg] == dim_name):
          return int(self.sizes[gid])
    return -1

  def get_portion_expression (self, pbs, proc_var):
    Nportion = ""
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      Nportion = '{}'.format (int(math.ceil (int(pbs) * 1.0 / int(proc_var))))
    else:
      Nportion = '({} / {})'.format (pbs, proc_var)
    return Nportion

  def set_block_function (self, mf, bvar, dim, pbs):
    cstr_sum = ""
    cstr_prod = ""
    block_sizes = []
    for pp in range(self.np):
      proc_var = self.PP.get_proc_dim_symbol (pp)
      pi_var = self.get_map_varname (dim, pp)
      Nportion = self.get_portion_expression (pbs, proc_var)
      if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
        block_sizes.append (int(Nportion))
      else:
        cstr_lb = '{} >= {} * {}'.format (bvar, Nportion, pi_var)
        self.cof.add_cstr (cstr_lb)
        cmd = 'opt.add ({})'.format (cstr_lb)
        self.writeln (mf, cmd)
      if (pp > 0):
        cstr_sum += " + "
        cstr_prod += " + "
      cstr_sum += '{} * {}'.format (Nportion, pi_var)
      cstr_prod += pi_var
    cstr_prod = self.get_sum_pi_var_along_dim (dim, -1)
    cstr = '{} == {} + {} - {} * ({})'.format (bvar, cstr_sum, pbs, pbs, cstr_prod)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    # Set the min block size with a constraint
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      min_block_size = min(block_sizes)
      cstr = '{} >= {}'.format (bvar, min_block_size)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
    # Modulo condition for selecting perfect multiples of the processor dimensions.
    if (USE_MODULO):
      for pp in range(self.np):
        proc_var = 'p{}'.format (pp)
        cstr = '{} % {} == 0'.format (pbs, proc_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  ## Declare variables with prefix 'DS_'
  ## Modified for FFT. Changed 'range(self.ndim)' to 'self.sizes'
  ## Reference
  def declare_block_variables (self, mf, decl):
    for dd in self.sizes:
      varname = self.get_block_size_var (dd)
      decl = self.declare_variable (mf, varname, decl)
      size = self.sizes[dd]
      self.set_upper_bound (mf, varname, size)
      self.set_block_function (mf, varname, dd, size)
    return decl

  ## Reference
  def is_dim_used (self, iter_dim_name):
    ## Modification for FFT.
    ## Check in lin_groups if there are singleton group. If so, use the index to access
    ## the dimension name.
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return True
    return False

  ## Reference: return a vector with 0s and 1s representing
  ## whether an iteration space dimension is used to access
  ## a reference.
  def get_vector_used_dims (self, stmt):
    ret=[0] * stmt.get_num_dim ()
    if (DEBUG_REF_USED_DIM):
      print ("stmt = {}, vec01 before = {}".format (stmt.get_name (), ret))
    for dd in range(self.ndim):
      iter_name = self.dims[dd]
      idim = stmt.get_dim_by_name (iter_name)
      ret[idim] = 1
    if (DEBUG_REF_USED_DIM):
      print ("==> vec01 after = {}".format (ret))
    return ret

  ## Reference
  ## get_dim_if_used 
  def get_dim_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return dd
    return -1

  def get_dim_size_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return int(self.sizes[dd])
    return 0

  ## FFT
  def is_dim_used_in_linear_group (self, iter_dim_name):
    for group in self.lin_groups:
      for gg in group:
        if (self.dims[gg] == iter_dim_name):
          return True
    return False

  ## FFT
  def get_dim_if_used_in_linear_group (self, iter_dim_name):
    for gid in range(len(self.lin_groups)):
      for gg in self.lin_groups[gid]:
        if (self.dims[gg] == iter_dim_name):
          return gid
    return -1


  ## @Reference:
  ## Return the pi mapping of the dimension corresponding
  ## to parameter iter_dim_name, if it is used.
  ## As the value -1 for a pi means that it is replicated,
  ## DIM_NOT_USED (-2) denotes 'not used'
  def get_pi_by_dim_name_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return DIM_NOT_USED

  def get_pi_by_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return DIM_NOT_USED

  def get_pi_dim_map (self, adim):
    if (adim >= len(self.map)):
      print ("[ERROR] Dimension requested does not exist ({} > {})".format (adim, len(self.map)))
      sys.exit (42)
    return self.map[adim]

  ## FFT
  ## Determine whether the rightmost dimension is a batched dimension.
  ## Assume that the batch dimension is identified with a suffix '#'. 
  ## The batch dimension can only be the last dimension of the FFT tensor.
  def has_batch_dim (self):
    found = False
    batch_dim = -1
    for dd in self.dims:
      if (self.dims[dd].find ('#') > 0):
        found = True
        batch_dim = dd
    last_dim = len(self.dims)-1
    if (found and batch_dim != last_dim):
      print ('[ERROR]: Found batch dim at ({}). If used, should be dimension {}'.format (batch_dim, last_dim))
      sys.exit (42)
    return found and batch_dim == last_dim

  ## Reference
  ## FFT
  def count_batch_dims (self):
    reserved = 0
    if (self.has_batch_dim ()):
      reserved += 1
    return reserved


  ## FFT
  def sum_pi_all (self):
    ret = ''
    for pp in range(self.PP.get_num_dim ()):
      pi_sum = self.get_sum_pi_var_along_dim (-1, pp)
      if (ret != ''):
        ret += ' + '
      ret += pi_sum
    return ret

  # Return the processor dimension associated to a data
  # space dimension by the name of the iterator used to access it.
  def get_proc_map_by_dim_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return -1


  def proc_map_match (self, iter_dim_name, ispace_pdim):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        ds_pdim = self.map[dd]
        if (ds_pdim == ispace_pdim):
          return True
    return False

  def all_pi_mu_match (self, stmt):
    for dd in self.dims:
      iter_name = self.dims[dd]
      stmt_pdim = stmt.get_proc_map_by_dim_name (iter_name)
      array_pdim = self.map[dd]
      if (stmt_pdim != array_pdim):
        return False
    return True


  ## Return the product of all extents of the array.
  def get_fixed_ub (self):
    ret = 1
    if (option_debug >= 2):
      print ("Sizes used by statement {}".format (self.name))
    for ss in self.sizes:
      if (option_debug >= 2):
        print ("{} ".format (self.sizes[ss]))
      ret *= int(self.sizes[ss])
    return ret
      
  ## Return the name of the capacity constraint for a given statement.
  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname

  def get_full_tensor_volume (self):
    full_vol = ''
    vol_var = 1
    for ee in self.sizes:
      if (full_vol != ''):
        full_vol += ' * '
      full_vol += str(self.sizes[ee])
      vol_var *= int(self.sizes[ee])
    return vol_var

  def get_max_extent (self):
    ret = 1
    for ee in self.sizes:
      ext = int(self.sizes[ee])
      if (ext > ret):
        ret = ext
    return ret

  ## Reference
  ## Tensor
  def get_fft_min_tensor_volume_expr (self, fft_stmt):
    full_vol = ''
    vol_var = 1
    for ee in self.sizes:
      if (full_vol != ''):
        full_vol += ' * '
      full_vol += str(self.sizes[ee])
      vol_var *= int(self.sizes[ee])
    eff_proc_count = self.get_effective_processor_count_in_batch (fft_stmt)
    ret = '({} / {})'.format (vol_var, eff_proc_count)
    return ret

  def build_memory_reduction_factor (self, is_comp_op = False):
    ret = ''
    factor_list = []
    for pp in range(self.PP.get_num_dim ()): 
      grid_var = self.PP.get_varname (pp)
      sum_pi = self.get_sum_pi_var_along_dim (-1, pp)
      factor = '({} + (1 - {}) * {})'.format (sum_pi, sum_pi, grid_var)
      factor_list.append (factor)
    ret = build_product_from_list (factor_list)
    if (not is_comp_op):
      return '(1)'
    if (self.PP.get_num_dim () == len(self.dims)):
      return '(1)'
    return ret

    
  ## Reference
  ## Declare volume variables (capacity variables) together with their defining 
  ## expressions. Volumes are computes from the block size associated to the 
  ## array and the dimensions being accessed.
  ## Expressions resulting are of the form: req_{aa} = \prod_{adim} DS_{aa,adim}
  def define_volume_var (self, mf, decl, is_comp_op = False):
    volvar = self.get_volume_var ()
    decl = self.declare_variable (mf, volvar, decl)
    prod_str = ""
    for dd in range(min(self.ndim,len(self.lin_groups))): # Minor modification to account for linearized tensors.
      if (dd > 0):
        prod_str += " * "
      varname = self.get_block_size_var (dd)
      prod_str += varname
      # Lower bound the
      cstr = '{} >= {}'.format (volvar, varname)
      cmd = 'opt.add ({}) ## Generated by define_volume_var'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    ## Alternate eqs below between == and >=
    is_parent_fft = self.parent_operator.is_fft ()
    fft_stmt = None
    if (is_parent_fft):
      fft_stmt = self.parent_operator
    eff_batch_vol = self.get_effective_processor_count_in_batch (fft_stmt)
    eff_batch_vol = self.get_fft_min_tensor_volume_expr (fft_stmt)
    cstr = '{} == {}'.format (volvar, prod_str)
    red_fact = self.build_memory_reduction_factor (is_comp_op)
    cstr = '{} == {} / ({})'.format (volvar, prod_str, red_fact)
    if (self.parent_operator == None):
      print ('[ERROR] Missing to set parent operator for Tensor {}'.format (self.name))
      sys.exit (42)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, '## Generated by define_volume_var. Parent ={}'.format (self.parent_operator.get_name ()))
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    ub = self.get_fixed_ub ()
    ## Volume var has a fixed upper bound.
    cstr = '{} <= {}'.format (volvar, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  # Return variable name of the form: LM_<stmt_name>_<array_name>
  ## In the paper, we refer to the 'LM' variables as '\lambda'.
  ## Variable to determine if access to an array is done in a local fashion.
  def get_match_variable (self, stmt_name):
    varname = 'LM_{}_{}'.format (stmt_name, self.name)
    return varname

  ## Local computation (local match) dimension variable.
  ## A boolean variable that represents if a computation is local.
  ## This happens when the array in question is not mapped to any
  ## processor dimension, or when both the mu and pi variables
  ## of the array and statement match.
  def get_match_dim_variable (self, stmt_name, dim):
    varname = 'LM_{}_{}_i{}'.format (stmt_name, self.name, dim)
    return varname

  ## Reference.
  def get_mu_variable (self, stmt_name, dd, pp):
    varname = 'mu_{}_i{}_p{}'.format (stmt_name, dd, pp)
    return varname

  ## Reference.
  def get_phi_variable (self, stmt_name, idim, adim, pdim):
    varname = 'phi_{}_{}_i{}_a{}_p{}'.format (stmt_name, self.name, idim, adim, pdim)
    return varname

  ## Reference.get_replication_comm_factor_variable ():
  def get_replication_comm_factor_variable (self, stmt_name):
    varname = 'Nrep_{}_{}'.format (stmt_name, self.name)
    return varname

  ## Reference.get_replication_comm_factor_variable ():
  def get_replication_comm_factor_dim_variable (self, stmt_name, pdim):
    varname = 'Nrep_{}_{}_p{}'.format (stmt_name, self.name, pdim)
    return varname

  ## Reference.get_replication_out_factor_expr (): Return the product
  ## expression of all Nrep_ variables.
  def get_replication_out_factor_expr (self, stmt_name):
    ret = ''
    for pp in range(self.PP.get_num_dim ()):
      if (pp > 0):
        ret += ' + '
      ret += self.get_replication_comm_factor_dim_variable (stmt_name, pp)
    return ret

  ## This function creates constraints of the form:
  ## LM_{stmt,ref} = (1 - sum pi_{stmt,ref,dim}) + sum pi_{stmt,ref_dim} x mu_{stmt,ref,dim}
  ## Each LM_{stmt,ref} is a boolean variable.
  ## In the paper we refer to the LM variables as lambda.
  def declare_matching_variables (self, mf, stmt_name, idim, sum_mu_var, dim_name, decl):
    # Must receive the array of mapped dimensions of the current statement.
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        uvar = self.get_match_dim_variable (stmt_name, idim)
        decl = self.declare_variable (mf, uvar, decl)
        self.set_bounds (mf, uvar, 0, 1)
        if (DIMAGE_USE_RHO_REPLICATION_FACTOR):
          N_rho_dim = self.get_replication_comm_factor_dim_variable (stmt_name, dd)
          if (not N_rho_dim in decl):
            decl = self.declare_variable (mf, N_rho_dim, decl)
            self.set_bounds (mf, N_rho_dim, 0, 1) 
        sum1 = ""
        sum2 = ""
        sum3 = ""
        # Factor to scale outgoing communication when tensor is replicated
        outfactor = "" 
        for pp in range(self.np):
          pivar = self.get_map_varname (dd, pp)
          muvar = self.get_mu_variable (stmt_name, idim, pp)
          proc_vardim_size = self.PP.get_varname (pp)
          match_term = '({} * {})'.format (muvar, pivar)
          # NOTE: We do not need this, after all.
          #local_cstr = '{} <= {}'.format (pivar, muvar)
          #self.add_constraint (mf, local_cstr)
          if (pp > 0):
            sum1 += " + "
            sum2 += " + "
            sum3 += " + "
            outfactor += " + "
          sum1 += pivar
          sum2 += match_term
          sum3 += '({}+{})%2'.format (muvar,pivar)
          outfactor += '{} * ({} - {})'.format (proc_vardim_size, muvar, pivar)
        ## Reminder: idim is the iteration space dimension, but within the
        ## reference it has another dimension position.
        ## Fetch the sum_pi variable for the given data-space dimension.
        sum1 = self.get_sum_pi_var_along_dim (dd, -1)
        match_expr = '{} == (1 - {} + {})'.format (uvar, sum1, sum2)
        self.add_constraint (mf, match_expr)
        if (DIMAGE_USE_RHO_REPLICATION_FACTOR):
          replication_expr = '{} >= (1 - {} + {})'.format (N_rho_dim, self.get_rho_varname (), outfactor)
          self.add_constraint (mf, replication_expr)
    return decl

  ## Reference.set_rho_var_dim_constraints ():
  ## Insert constraints to activate the rho_dim_var depending on whether its 
  ## a data replication scenario or a reduction scenario.
  def set_rho_var_dim_constraints (self, mf, decl, stmt):
    for pp in range(self.PP.get_num_dim ()):
      N_rho_dim = self.get_replication_comm_factor_dim_variable (stmt.get_name (), pp)
      if (not N_rho_dim in decl):
        decl = self.declare_variable (mf, N_rho_dim, decl)
        self.set_bounds (mf, N_rho_dim, 0, 1) 
        cstr = '{} >= 1 - {}'.format (N_rho_dim, self.get_sum_pi_var_along_dim (-1, pp))
        self.add_constraint (mf, cstr)
        red_expr = stmt.get_sum_reduction_mu_expr_along_dim (self, pp)
        if (red_expr != ''):
          cstr = '{} >= ({})'.format (N_rho_dim, red_expr)
          self.add_constraint (mf, cstr)
    return decl


  # One boolean variable per stmt, per array
  def declare_matching_variables_with_phi (self, mf, stmt_name, idim, dim_name, decl):
    # Must receive the array of mapped dimensions of the current statement.
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        uvar = self.get_match_dim_variable (stmt_name, idim)
        decl = self.declare_variable (mf, uvar, decl)
        self.set_bounds (mf, uvar, 0, 1)
        sum1 = ""
        sum2 = ""
        for pp in range(self.np):
          pivar = self.get_map_varname (dd, pp)
          muvar = self.get_mu_variable (stmt_name, idim, pp)
          phivar = self.get_phi_variable (stmt_name, idim, dd, pp)
          decl = self.declare_variable (mf, phivar, decl)
          self.set_bounds (mf, phivar, 0, 1)
          self.add_constraint (mf, '{} <= {}'.format (phivar, pivar))
          self.add_constraint (mf, '{} <= {}'.format (phivar, muvar))
          match_term = phivar
          if (pp > 0):
            sum1 += " + "
            sum2 += " + "
          sum1 += pivar
          sum2 += match_term
        match_expr = '{} == (1 - {} + {})'.format (uvar, sum1, sum2)
        self.add_constraint (mf, match_expr)
    return decl

  # Return *READ* communication variable for statement name given.
  def get_stmt_read_ref_comm_var (self, stmt_name):
    varname = 'ReadK_{}_{}'.format (stmt_name, self.name)
    return varname

  # Return *WRITE* communication variable for statement name given.
  def get_stmt_write_ref_comm_var (self, stmt_name):
    varname = 'WriteK_{}_{}'.format (stmt_name, self.name)
    return varname

  # Declare communication varible K_{stmt}_{array}
  def define_stmt_ref_comm_var (self, mf, stmt_name, decl):
    commvar = self.get_stmt_read_ref_comm_var (stmt_name)
    decl = self.declare_variable (mf, commvar, decl)
    return decl

  # Return the name of the variable representing if an array slice is
  # locally mapped.
  def get_local_ref_vol_var (self, stmt_name):
    varname = 'Local_{}_{}'.format (stmt_name, self.name)
    return varname

  def define_stmt_ref_local_vol_var (self, mf, stmt_name, decl):
    localvar = self.get_local_ref_vol_var (stmt_name)
    decl = self.declare_variable (mf, localvar, decl)
    return decl

  def get_local_ref_dim_vol_var (self, stmt_name, idim):
    varname = 'Local_{}_{}_i{}'.format (stmt_name, self.name, idim)
    return varname

  def define_stmt_ref_dim_local_vol_var (self, mf, stmt_name, idim, decl):
    localvar = self.get_local_ref_dim_vol_var (stmt_name, idim)
    decl = self.declare_variable (mf, localvar, decl)
    return decl

  def tensor_extract_dims_from_pi_var (self, pi_var):
    parts = pi_var.split ("_")
    idim_str = re.sub ("i","",parts[2])
    idim = int(idim_str)
    pdim_str = re.sub ("p","",parts[3])
    pdim = int(pdim_str)
    return (idim,pdim)

  ## Reference.
  ## Extract the pi mappings from the solution set and store 
  ## them in the map attribute.
  def extract_mappings_from_solution_set (self, solset):
    for vv in solset:
      if (vv.find ("sum") == 0):
        continue
      piprefix='pi_{}_'.format (self.name)
      if (vv.find (piprefix) == 0):
        if (int(solset[vv]) == 1):
          idim, pdim = self.tensor_extract_dims_from_pi_var (vv)
          self.map[idim] = pdim

  def prep_dist_vec_for_map_key (self):
    alist = []
    n_dim = len(self.dims)
    for adim in range(len(self.sizes)):
      onezero = [0] * self.PP.get_num_dim ()
      pi_map = self.map[adim]
      if (pi_map >= 0):
        onezero[pi_map] = 1
      key01 = build_string_from_list (onezero, '')
      alist.append (key01)
    return alist

  def get_map_key (self):
    alist = self.prep_dist_vec_for_map_key ()
    ret = self.get_name () + '_' + build_string_from_list (alist, 'x')
    return ret

  def get_rotated_map_key (self):
    km = self.prep_dist_vec_for_map_key ()
    ret = self.get_name () + '_' + build_string_from_list (alist, 'x')
    return ret

  def get_udim_varname (self, stmt_name):
    varname = 'DIMAGE_UDIM_{}_{}'.format (stmt_name, self.name)
    return varname

  def get_amap_varname (self):
    varname = 'DIMAGE_AMAP_{}'.format (self.name)
    return varname

  ## Generate a global array declaration finalized with a '-2'
  def generate_ref_amap_declarations (self, mf):
    dimlist = ""
    for dd in self.map:
      dimlist += '{}'.format (self.map[dd])
      dimlist += ', '
    dimlist += '-2'
    varname = self.get_amap_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def indent (self, df):
    df.write ('  ')

  def is_map_transposed (self):
    pdim = []
    for dd in self.dims:
      if (self.map[dd] >= 0):
        pdim.append (self.map[dd])
    if (len(pdim) <= 1):
      return False
    if (len(pdim) == 2):
      if (pdim[0] > pdim[1]):
        return True
      return False
    print ("Unhandled case. Will abort.")
    sys.exit (42)
    return False
      

    print ('WTF WHY: len={}'.format (len(self.map)))
    sys.exit (42)
    return False

  ## @Reference
  def generate_communicators_at_statement (self, df, stmt):
    comm_var = self.get_array_communicator_at_statement (stmt.get_name ())
    self.indent (df)
    df.write ('// Compute color for communicator #{} used by array [{}] at statement {}()\n'.format (comm_var, self.name, stmt.get_name ()))
    rank = DIMAGE_PROC_RANK
    udim_var = self.get_udim_varname (stmt.get_name ())
    imap_var = stmt.get_imap_varname ()
    amap_var = self.get_amap_varname ()
    commvec = 'comm_vec'
    comm_variant = ''
    if (not stmt.is_data_sink () and stmt.is_output_array (self)):
      comm_variant = 'generator_'
    is_trans = 0
    if (self.is_map_transposed ()):
      is_trans = 1
    line = 'compute_{}comm_vector ({}, {}, {}, {}, {}, {});\n'.format (comm_variant, rank, udim_var, imap_var, amap_var, commvec, is_trans)
    self.indent (df)
    df.write (line)
    comm_color = 'comm_color'
    nprocdim = self.np
    # Build call to dimage_compute_color_from_comm_vec.
    line = '{} = {} ({}, {}, {}, {});\n'.format (comm_color, DIMAGE_COMPUTE_COLOR_FUNC, nprocdim, DIMAGE_GRID_DIMS, DIMAGE_PROC_COORDS, commvec)
    self.indent (df)
    df.write (line)
    line = 'MPI_Comm_split (MPI_COMM_WORLD, {}, {}, &{});\n'.format (comm_color, rank, comm_var)
    self.indent (df)
    df.write (line)
    self.indent (df)
    df.write ('log_num("Communicator {} color", {});\n'.format (comm_var,comm_color))
    self.indent (df)
    df.write ('log_commvec("Comm.vector {} ", {}, {});\n'.format (commvec,commvec,self.np))

  def declare_communicator_at_statement (self, df, stmt):
    comm_var = self.get_array_communicator_at_statement (stmt.get_name ())
    df.write ('MPI_Comm {};\n'.format (comm_var))

  # Reference.get_num_proc_along_dim_at_current ():
  # Return the number of processors along the data space dimension @dd.
  # If the data dimension is unmapped, return '1' since we assume it's a local
  # computation.
  def get_num_proc_along_dim_at_current (self, dd, PP):
    if (dd >= len(self.map)):
      print ('[ERROR@get_num_proc_along_dim]: Invalid dimension requested.')
      sys.exit (42)
    pdim = self.map[dd]   
    if (pdim < 0):
      return '1'
    return str(PP.get_dim_size (pdim))
    
  # Return the dimension size, possibly tiled, of the current array and 
  # dimension. Will take into account the number of processors and the processor
  # geometry.
  # NOTE: the @dd argument must be an index in the array reference, not the
  # statement.
  def get_dimension_size_as_str (self, stmt, dd, PP, alloc_mode):
    return "N/A"

  def get_full_dimension_size_as_str (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    return '{}'.format (num)


  def get_num_proc_along_dim (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    pdim = self.map[dd]
    iter_name = self.dims[dd]
    denum = 1
    if (stmt != None):  # -1 == unmapped
      # Statement could still access in a tiled fashion.
      # Check if statement is mapped at dimension @dd.
      stmt_idim = stmt.get_dim_by_name (iter_name)
      if (stmt_idim >= 0):
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        if (option_debug >= 2):
          print ("\t[INFO@get_num_proc_along_dim]: producer {} array {}, dimension {} : map = {}".format (stmt.get_name (), self.name, iter_name, stmt_pdim))
        if (stmt_pdim >= 0):
          denum = PP.get_dim_size (stmt_pdim)
    if (pdim >= 0):
      denum = max(denum, PP.get_dim_size (pdim))
    return str(denum)

  ## Return an expression representing the data slice of an array dimension.
  ## Original array extent is divided by the number of processors if mapped.
  ## Combinations of mu and pi mappings are considered.
  def get_dimension_size_as_val (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    pdim = self.map[dd]
    iter_name = self.dims[dd]
    denum = 1
    tag = 'dsav1'
    if (stmt != None and pdim == -1):  # -1 == unmapped
      # Statement could still access in a tiled fashion.
      # Check if statement is mapped at dimension @dd.
      stmt_idim = stmt.get_dim_by_name (iter_name)
      if (stmt_idim >= 0):
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        if (stmt_pdim >= 0):
          denum = PP.get_dim_size (stmt_pdim)
          tag = 'dsav2'
        else:
          denum = 1
          tag = 'dsav3'
    if (pdim >= 0):
      denum = PP.get_dim_size (pdim)
      tag = 'dim-size'
    dim_size = '{} (({}), {} /* {} */)'.format (DIMAGE_CEIL, num, denum, tag)
    if (int(denum) == 1):
      dim_size = '{}'.format (num)
    return dim_size

  ## Return the number of tiles (blocks) along a data dimension (@dd).
  def reference_get_num_mapped_tiles_at_dim (self, stmt, dd):
    return 99999999
    
  ## Reference.get_aggregated_tile_header_space(): Compute the aggregated 
  ## payload associated to all tile headers.
  def get_aggregated_tile_header_space (self, stmt):
    ret = str(DIMAGE_TILE_HEADER_SIZE)
    for dd in self.dims:
      ret += ' * '
      ret += str(self.reference_get_num_mapped_tiles_at_dim (stmt, dd))
    return ret

  def get_dimension_size_as_str_list (self, stmt, PP, alloc_mode):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      if (alloc_mode == ALLOC_MODE_FULL):
        ret += self.get_full_dimension_size_as_str (stmt, dd, PP)
      else:
        ret += self.get_dimension_size_as_str (stmt, dd, PP, alloc_mode)
    return ret

  ## Return the array extent give a dimension.
  def get_extent_as_str (self, dd):
    num = '{}'.format (self.sizes[dd])
    return num

  def get_extent_as_str_by_dim_name (self, iter_name):
    for dd in self.sizes:
      if (self.dims[dd] == iter_name):
        return self.get_extent_as_str (dd)
    return "ERROR"

  ## @Reference: Return a list of array extents separated by commas.
  def get_array_extents_as_str_list (self):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      ret += self.get_extent_as_str (dd)
    return ret

  ## Return the array extent give a dimension.
  ## This method is exclusively used by get_mapped_array_extents_as_str_list.
  def get_mapped_extent_as_str (self, dd, use_full, ext_mu):
    if (dd >= len(self.dims)):
      print ("ERROR: Requested array dimension {}, but array only has {} dimensions.".format (dd, len(self.dims)))
      sys.exit (42)
    pi_dim = self.map[dd]
    ret = '{}'.format (self.sizes[dd])
    if (pi_dim >= 0 and not use_full):
      denum = self.PP.get_dim_size (pi_dim)
      if (int(denum) > 1):
        ret = '{}({},{})'.format (DIMAGE_CEIL,ret,denum)
    elif (ext_mu >= 0 and not use_full):
      denum = self.PP.get_dim_size (ext_mu)
      if (int(denum) > 1):
        ret = '{}({},{})'.format (DIMAGE_CEIL,ret,denum)
    return ret

  ## @Reference: Return a list of mapped array extents separated by commas.
  ## Returned extents are the original extents divided by the number
  ## of processors along their mapped dimension.
  ## Argument ext_mu determines if we are accessing a local
  ## buffer and not a 'frozen' layout. If ext_mu == -1, then we ignore it.
  def get_mapped_array_extents_as_str_list (self, stmt = None, use_full = True, is_write = False):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      ext_mu = DIM_UNMAPPED
      if (stmt != None and is_write):
        iter_name = self.dims[dd]
        ext_mu = stmt.get_mu_dim_map_by_name (iter_name)
      ret += self.get_mapped_extent_as_str (dd, use_full, ext_mu)
    return ret

  def get_array_size_as_product_str (self):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ' * '
      ret += self.get_extent_as_str (dd)
    return ret

  ## Reference.get_tile_allocator_name ():
  def get_tile_allocator_name (self):
    allocator = ''
    if (len(self.dims) == 1):
      allocator = DIMAGE_TILE_ALLOCATOR_1D
    if (len(self.dims) == 2):
      allocator = DIMAGE_TILE_ALLOCATOR_2D
    if (len(self.dims) == 3):
      allocator = DIMAGE_TILE_ALLOCATOR_3D
    if (len(self.dims) == 4):
      allocator = DIMAGE_TILE_ALLOCATOR_4D
    return allocator

  def get_single_node_tile_coord (self):
    ret = ''
    for ii in range(len(self.dims)):
      if (ii > 0):
        ret += ', '
      ret += '1'
    return ret
    

  ## Reference.allocate_local_tile (): create a call to an N-dimensional
  ## tile allocator.
  ## This method is called only from Statement.generate_operator ().
  def allocate_local_tile (self, df, PP, is_generator, stmt, is_single_node = False):
    allocator = self.get_tile_allocator_name ()
    if (allocator == ''):
      print ('[ERROR]: Unsupported tile dimension (> 4D)')
      sys.exit ()
    stride_list = self.get_tile_extent_list ()
    proc_geom = PP.get_processor_geometry_list_from_map (stmt.get_mu_map (), False)
    if (is_generator and stmt != None):
      proc_geom = PP.get_processor_geometry_list_from_map (self.map, False)
    buffer_name = self.name
    if (not is_generator):
      buffer_name = self.get_name_for_check () 
    if (is_single_node):
      # Adjust the buffer name and processor geometry for use on a single node.
      buffer_name = self.get_sna_ref_name ()
      proc_geom = self.get_single_node_tile_coord ()
    line = '{} = {}({}, {});\n'.format (buffer_name, allocator, stride_list, proc_geom)
    df.write ('\n')
    self.indent (df)
    df.write (line)

  ## Reference.allocate_tile_map(): Allocate memory for a local tile.
  ## Meant to be used in generators only.
  def allocate_tile_map (self, df, PP, ext_buffer = None):
    varname = self.get_tile_map_name (ext_buffer)
    arrdim = len(self.dims)
    tile_shape = PP.get_processor_geometry_list_from_map (self.map, True)
    line = '{} = {}_{}D ({});\n'.format (varname, DIMAGE_TILE_MAP_ALLOCATOR, arrdim, tile_shape)
    df.write ('\n')
    self.indent (df)
    df.write (line)
    
    

  ## @Reference: store the generated tile for debug.
  def dump_generated_tile (self, df, PP):
    dimensions = self.get_dimension_size_as_str_list (None, PP, ALLOC_MODE_TILE)
    pclist = PP.get_processor_coordinate_str_list ()

    strides = self.get_tile_extent_list ()
    proc_geom = self.PP.get_processor_geometry_list_from_map (self.get_pi_map (), False)

    line = '{}_tile{}D ("{}", {}, {}, {}, {}); /* write-to-file arg */\n'.format (WRITE_TO_FILE_FUNC, self.ndim, self.name, DIMAGE_RANK_ARRAY, self.name, strides, proc_geom)
    
    df.write ('\n')
    self.indent (df)
    df.write ('#ifdef DIMAGE_DEBUG\n')
    self.indent (df)
    df.write (line)
    self.indent (df)
    df.write ('#endif\n')

  def return_allocated (self, df):
    line = 'return {};\n'.format (self.name)
    self.indent (df)
    df.write (line)

  ## @Reference
  def reference_get_local_volume (self, stmt):
    ret = ""
    header = ""
    for dd in self.sizes:
      if (not ret == ""):
        ret += " * " 
        header += " * "
      lexval = self.get_dimension_size_as_val (stmt, dd, PP)
      tiles = self.reference_get_num_mapped_tiles_at_dim (stmt, dd)
      ret += lexval
      header += str(tiles)
    ret = '{} + {} * {}'.format (ret, DIMAGE_TILE_HEADER_SIZE, header)
    return ret

  def get_full_volume (self, stmt):
    ret = ""
    for dd in self.sizes:
      if (ret != ""):
        ret += " * " 
      ret += self.sizes[dd]
    return ret

  def get_tile_vol (self, stmt):
    return self.reference_get_local_volume (stmt)


  ## @Reference.get_tile_extent_list (): Return a comma-separated list of
  ## tile extent dimensions.
  def get_tile_extent_list (self):
    ret = ""
    for dd in self.sizes:
      if (not ret == ""):
        ret += ", " 
      num = int(self.sizes[dd])
      pdim = self.map[dd]
      denum = 1 #self.PP.lcm ()
      extent = '{} (({}), {})'.format (DIMAGE_CEIL, num, denum)
      ret += extent
    return ret

  def get_slice_varname (self, as_in):
    varname = 'slice_{}'.format (self.name)
    if (as_in):
      varname = 'ra_{}'.format (varname)
    else: 
      varname = 'wa_{}'.format (varname)
    return varname

  def set_use_slice (self, val):
    self.uses_slice = val

  def get_use_slice (self):
    return self.uses_slice

  ## Reference.generate_local_slice_buffer(): Generate the code to declare and
  ## allocate a slice of data tiles.
  def generate_local_slice_buffer (self, df, slice_vol, as_in):
    slice_var = self.get_slice_varname (as_in)
    allocator = self.get_tile_allocator_name ()
    line = '{} * {} = {}({});\n'.format (DIMAGE_DT, slice_var, allocator, slice_vol)
    self.indent (df)
    df.write (line)
    return slice_var

  ## Reference.generate_incoming_communication ():
  ## Determine the set of references of a given operator (@stmt), that
  ## requires incoming communication.
  ## For each ref used in @stmt, we check its pi-mapping relative to
  ## the mu-mapping of @stmt.
  ## When communication is found to be necessary we generate (declare) the
  ## necessary buffers. 
  ## The function returns the variable name generated or None if no 
  ## communication is deemed required.
  def reference_generate_incoming_communication (self, df, stmt, comm_type, PP):
    self.indent (df)
    df.write ("// Info of array {}[]\n".format (self.name))
    send_size = self.reference_get_local_volume (stmt)
    recv_size = stmt.get_slice_vol_by_name (self, PP)
    local_vol = self.reference_get_local_volume (stmt)
    self.indent (df)
    df.write ("// Local volume (elements): {}\n".format (local_vol))
    self.indent (df)
    df.write ("// Recv volume  (elements): {}\n".format (recv_size))
    self.indent (df)
    df.write ("// Comm. type: {}\n".format (comm_type_str (comm_type)))
    communicator = self.get_array_communicator_at_statement (stmt.get_name ())
    self.indent (df)
    df.write ("// Array communicator @ {}: {}\n".format (stmt.get_name(), communicator))
    generated_slice = None
    if (comm_type == COMM_TYPE_LOCAL):
      self.indent (df)
      df.write ("// Nothing to do\n")
    if (comm_type == COMM_TYPE_GATHER_SLICE):
      ## NOTE: Only case of incoming all-gather.
      self.set_use_slice (True)
      stride_list = self.get_tile_extent_list ()
      vec01 = self.get_vector_used_dims (stmt)
      proc_geom = PP.get_processor_geometry_list_from_map (stmt.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* in-alloc-args */'.format (stride_list, proc_geom)
      slice_var = self.generate_local_slice_buffer (df, alloc_args, True)
      generated_slice = slice_var
      primitive = COLLECTIVE_ALLGATHER
      # current array is only a piece of the slice.
      source_array = self.name 
      target_array = slice_var
      collective = '{} ({}, {}, {}, {}, {}, {}, {});\n'.format (primitive, source_array, send_size, get_mpi_datatype(DIMAGE_DT), target_array, send_size, get_mpi_datatype(DIMAGE_DT), communicator)
      self.indent (df)
      df.write (collective)
      ## Since we do an all-gather, we need to build the *new* map of tile blocks.
      self.set_is_allgat_in_slice (True)
      intermediate = None
      if (self.get_use_slice ()):
        intermediate = self.get_slice_varname (True)
      self.indent (df)
      df.write ('// Rebuild tile map after all-gather on buffer {}\n'.format (intermediate))
      self.indent (df)
      tile_map_name = self.get_tile_map_name (intermediate)
      line = '{} * {};'.format (DIMAGE_INT, tile_map_name)
      df.write (line)
      self.indent (df)
      self.allocate_tile_map (df, PP, intermediate)
      self.generate_tile_map_creation_code (df, PP, intermediate, proc_geom)
      df.write ('\n')
      self.append_to_free_list (tile_map_name)
      self.append_to_free_list (intermediate)
    if (comm_type == COMM_TYPE_P2P):
      print ('[ERROR@reference_generate_incoming_communication]: Unexpected P2P comm.type')
      sys.exit (42)
    return generated_slice

  # @Reference.generate_outgoing_communication(): Invoke only for non-sink statements.
  def generate_outgoing_communication (self, df, stmt, comm_type, PP):
    send_size = self.reference_get_local_volume (stmt)
    recv_size = stmt.get_slice_vol_by_name (self, PP)
    is_gen = stmt.is_data_generator ()
    local_vol = self.reference_get_local_volume (stmt)
    if (option_debug >= 2):
      print ("[INFO@generate_outgoing_communication] COMM.TYPE for array {} @ stmt {}: {}".format (self.name, stmt.get_name (), comm_type_str(comm_type)))
    self.indent (df)
    df.write ("// Local volume (elements): {}\n".format (local_vol))
    self.indent (df)
    df.write ("// Recv volume  (elements): {}\n".format (recv_size))
    self.indent (df)
    df.write ("// Comm. type: {}\n".format (comm_type_str (comm_type)))
    if (comm_type == COMM_TYPE_LOCAL_SLICE and stmt.is_true_communication (self)):
      self.indent (df)
      df.write ("// Computation is local, but local contribution must be reconciled across the whole slice. AllGather will follow\n")
      slice_var = self.get_slice_varname (False)
      recv_buff = self.name
      header_size = self.get_tile_header_size (False)
      if (not is_gen): # If @stmt is not a generator, we need an intermediate buffer for the allgather.
        recv_buff = stmt.generate_intermediate_allred_buffer (df, self)
      communicator = self.get_array_communicator_at_statement (stmt.get_name ())
      primitive = COLLECTIVE_ALLGATHER
      collective = '{} ({}, {}, {}, {}, {}, {}, {});\n'.format (primitive, slice_var, send_size, get_mpi_datatype(DIMAGE_DT), recv_buff, send_size, get_mpi_datatype(DIMAGE_DT), communicator)
      self.indent (df)
      df.write (collective)
    if (comm_type == COMM_TYPE_ALLRED and stmt.is_true_communication (self)):
      self.indent (df)
      df.write ("// Reduction dimension was mapped. AllReduce will follow\n")
      read_slice_var = self.get_slice_varname (False)
      interm = stmt.generate_intermediate_allred_buffer (df, self)
      header_size = self.get_tile_header_size (False)
      communicator = self.get_array_communicator_at_statement (stmt.get_name ())
      primitive = COLLECTIVE_ALLREDUCE
      send_size = recv_size
      collective = '{} ({}, {}, {} + {}, {}, {}, {});\n'.format (primitive, read_slice_var, interm, send_size, header_size, get_mpi_datatype(DIMAGE_DT), REDUCE_OP_ADD, communicator)
      self.indent (df)
      df.write (collective)

  ## Reference.generate_tile_map_creation_code ():
  def generate_tile_map_creation_code (self, df, PP, ext_buffer = None, ext_tile_list = None):
    tile_map = self.get_tile_map_name (ext_buffer)
    stride_list = self.get_tile_extent_list ()
    max_tile_list = PP.get_processor_geometry_list_from_map (self.map, True)
    ref_tile_list = PP.get_processor_geometry_list_from_map (self.map, False)
    arrdim = len(self.dims)
    buff_name = self.name
    if (ext_buffer != None):
      buff_name = ext_buffer
      ref_tile_list = ext_tile_list
    collect_call = '{}_{}D ({}, {}, {}, {}, {});\n'.format (DIMAGE_COLLECT_TILE_MAP_FUNC, arrdim, buff_name, tile_map, stride_list, max_tile_list, ref_tile_list)
    self.indent (df)
    df.write (collect_call)
    store_call = '{}_{}D (\"{}\", {}, {}, {});\n'.format (DIMAGE_STORE_TILE_MAP_FUNC, arrdim, tile_map, DIMAGE_RANK_ARRAY, tile_map, max_tile_list)
    self.indent (df)
    df.write ('#ifdef DIMAGE_DEBUG\n')
    self.indent (df)
    df.write (store_call)
    self.indent (df)
    df.write ('#endif\n')

  def append_to_free_list (self, varname):
    self.free_list += '  if ({} != NULL) free ({});\n'.format (varname, varname)

  def get_free_list (self):
    return self.free_list

  def has_unmapped_dim (self, stmt):
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (self.map[dd] < 0):
        return True
    return False

  # @Reference: Return the access type for a single array reference (the @self).
  # Cases to consider:
  # Iter.Space x Data Space:
  #   M x M : tiled
  #   M x U : Sliced
  #   U x M : Even possible?
  #   U x U : tiled (full matrix)
  # A single data space dimension with different mapping suffices
  # to qualify the entire reference.
  def get_access_type (self, stmt, PP, producers):
    if (not self.name in producers):
      print ("Producers: {}".format (producers))
    is_lexpos = producers[self.name].is_mu_lexico_positive (self, producers)
    was_allgat =  producers[self.name].was_allgathered (self, stmt)
    return ACC_TYPE_TILE
    print ('[INFO@get_access_type: statement {}, reference {}, access {}, dmap={}, imap={}, lexpos={}, was_allgat={}'.
      format (stmt.get_name (), self.name, self.dims, self.map, stmt.get_mu_map (), is_lexpos, was_allgat))
    if (not is_lexpos and was_allgat): 
      # non lexico-pos partition and reassembling with allgather => layout change
      return ACC_TYPE_SLICE
    if (self.all_pi_mu_match (stmt)): # processor only ever owned a single tile and perfectly matched
      return ACC_TYPE_TILE
    if (is_lexpos and was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    if (is_lexpos and not was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    if (not is_lexpos and not was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    print ('[ERROR@get_access_type: Shouldn\'t get here.')
    return ACC_TYPE_ERROR

  # @Reference: Produce a linearized iterator list in string format.
  # Linearized access is only used when the array is deemed to have a 
  # lexico non-negative layout, e.g. <0,*> or <*,*>. 
  # As a result, iterators will not need to be reordered as in the
  # SLICE_ND access case.
  def get_linearized_iterator_str_list (self, stmt, PP, producers, is_write, is_acum):
    ttc = stmt.collect_tile_trip_counts (producers)
    ret = ''
    for adim in self.dims:
      iter_name = self.dims[adim]
      stmt_iter_idx = stmt.get_dim_used_in_ref_dim (self, adim)
      # Next, determine if a tile iterator is 'local'. If that's the case,
      # the data space dimension is effectively distributed, and the tile 
      # index offset associated to the dimension becomes zero. We enforce
      # this by multiplying by a zero factor.
      mu_map = stmt.get_mu_map ()
      mu_pdim = mu_map[stmt_iter_idx]
      pi_pdim = self.map[adim]
      factor = ''
      match_dim = self.is_pi_map_dim_equal (iter_name, mu_pdim)
      comm_type = stmt.determine_communication_type (self, PP)
      if (match_dim):
        factor = ' * 0'
      elif (mu_pdim >= 0 and pi_pdim < 0 and comm_type == COMM_TYPE_LOCAL_SLICE):
        if (not is_acum and is_write):
          ## if is_acum = False means it's part of the main computation
          ## is_write tells us is the slice-buf
          factor = ' * 0 /* pi < 0, slice-buffer -> 0 (ct={})*/'.format (comm_type)       
        elif (is_acum):
          factor = ' * 1 /* PROB: was 0, {}*/'.format (comm_type)
        else:
          factor = ' * 1 /* CHECK ME */'.format (comm_type)
      if (ret != ''):
        ret += ', '
      tile_iter = stmt.get_tile_iterator (stmt_iter_idx) + factor
      point_iter = stmt.get_point_iterator (stmt_iter_idx)
      extent = self.get_extent_as_str (adim)
      num_proc_along_dim = '1'
      if (iter_name in ttc):
        num_proc_along_dim = ttc[iter_name]
      stmt_pmap = stmt.get_proc_dim_map (stmt_iter_idx)
      if (stmt_pmap >= 0):
        num_proc_along_dim = PP.get_dim_size (stmt_pmap)
      tile_size = '{}(({}),{})'.format (DIMAGE_CEIL, extent, num_proc_along_dim)
      expr = '({}) * ({}) + ({})'.format (tile_iter, tile_size, point_iter)
      ret += expr
    return ret

  def get_right_stride (self, adim):
    ret = ''
    for dd in range(adim+1,len(self.dims)):
      ret += ' * {}'.format (self.sizes[dd])
    return ret

  def gen_canonical_access (self):
    ret = ''
    ret += self.name
    ret += '['
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (dd > 0):
        ret += ' + '
      stride = self.get_right_stride (dd)
      ret += '{}{}'.format (iter_name, stride)
    ret += ']'
    return ret


## Start of Statement Class
class Statement:
  def __init__(self, form, PP, NP):
    self.name = ""
    self.PP = PP
    self.np = NP
    self.cof = form
    self.ndim = 0
    self.dims = {}
    self.nref = 0
    self.refs = {}
    self.accs = [] # same as refs but a list
    self.map = {}
    self.last_writer_map = None
    self.ntpd = []
    self.kernel_name = None
    self.linop = False
    self.optype = DIMAGE_OP_TYPE_LINALG

  def operator_init_from_file (self, ff, header):
    line = header
    line = line.strip ()
    # Expect two or three parts: <name>:<iterators>:<kernel_name>
    parts = line.split (':')
    self.name = parts[0]
    # Expect iterators of dimensions separated by ','
    dimlist = parts[1].split (RELS_LIST_SEPARATOR)
    if (len(parts) == 3):
      self.kernel_name = parts[2]
      if (self.kernel_name == 'fft'):
        self.optype = DIMAGE_OP_TYPE_FFT
      if (self.kernel_name == 'lin'):
        self.optype = DIMAGE_OP_TYPE_LINEARIZER
      print ("Kernel name found: {}".format (self.kernel_name))
    for dd,dname in enumerate(dimlist):
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    line = ff.readline ()
    line = line.strip ()
    self.nref = int(line)
    for aa in range(self.nref):
      ref = Reference (self.cof, self.PP, self.np)
      ref.tensor_init_from_file (ff)
      ref.set_parent_operator (self)
      self.refs[ref.get_name ()] = ref
      self.accs.append (ref)
    
  def estimate_memory_requirements (self):
    total = 0
    print ("Mem-req Statement : {}".format (self.name))
    for ref in self.accs:
      ref_req = ref.estimate_memory_requirements () 
      total += ref_req
      print ("{} --> {}".format (ref.get_name (), ref_req))
    return total

  def check_capacity_requirement (self, solset, pnc):
    volvar = self.get_volume_var ()
    for vv in solset:
      if (vv.find (volvar) >= 0):
        used = float(solset[vv])
        if (pnc > 0):
          used = used * 100 / pnc
        print ('Memory used by {} : {:.2f}% (of {})'.format (self.name, used, pnc))


  ## Operator.
  def get_linearizer_objective_var (self):
    return 'G_{}_lin'.format (self.name)


  def is_linearizer (self):
    ret = self.optype == DIMAGE_OP_TYPE_LINEARIZER
    if (ret):
      if (len(self.accs) != 2):
        print ("[ERROR] Linearizer operators should only access two tensors.")
        sys.exit (42)
    return ret

  def is_fft (self):
    return self.optype == DIMAGE_OP_TYPE_FFT

  def is_linalg (self):
    return self.optype == DIMAGE_OP_TYPE_LINALG

  def init_FFT (self):
    return

  ## FFT
  ## Connect pi vars of linearized tensor to pi vars of tensor to linearize.
  ## Do this for each dimension of the grid of PEs.
  def linearizer_connect_in_out_pi_vars (self, mf):
    print ('Connecting pi dimensions ...')
    out_dims = self.accs[1].dims
    for idx,do in enumerate(out_dims):
      in_dims = out_dims[do].split ('*')
      for pp in range(self.PP.get_num_dim ()):
        tgt_pi_var = self.accs[1].get_map_varname (do, pp)
        line = ''
        line_pi_vars = ''
        expr = ''
        for di in in_dims:
          line += ' : '
          src_pi_var = self.accs[0].get_pi_var_by_dim_name (di, pp)
          line += str(di)
          line_pi_vars += ' : '
          line_pi_vars += src_pi_var
          if (expr != ''):
            expr += ' * '
          expr += src_pi_var
        cstr = '{} == {}'.format (tgt_pi_var, expr)
        self.add_constraint (mf, cstr, ' linearizer contraint ')


  ## Statement.writes_to (): return True if the current statement
  ## modified the given array. Return False otherwise.
  def writes_to (self, ref):
    if (self.is_data_sink ()):
      return False
    nref = len(self.accs)
    write_ref = self.accs[nref-1]
    return (write_ref.get_name () == ref.get_name ())


  def set_last_writer_map (self, arg_last_writer_map):
    self.last_writer_map = arg_last_writer_map
    
  def get_ref (self, refid):
    if (refid >= len(self.accs)):
      return None
    return self.accs[refid]

  def get_ref_by_name (self, ref_name):
    for ref in self.accs:
      if (ref.get_name () == ref_name):
        return ref
    return None

  def gen_matrix_data (self):
    if (self.is_data_generator ()):
      self.accs[0].gen_matrix_data ()

  ## Statement.
  def show_info(self):
    print ("Statement: {} (ndim={})".format (self.name, self.ndim))
    for dd in self.dims:
      print ("Dim {}: {}".format (dd, self.dims[dd]))
    for aa in self.refs:
      self.refs[aa].show_info ()

  def get_dims (self):
    return self.dims

  def get_mu_map (self):
    return self.map

  def get_mu_dim_map (self, idim):
    if (idim >= len(self.map)):
      print ("ERROR @ get_mu_map")
      sys.exit (42)
    return self.map[idim]

  def get_mu_dim_map_by_name (self, dim_name):
    for ii in self.dims:
      if (self.dims[ii] == dim_name):
        return self.map[ii]
    return 'ERROR'

  def show_maps (self):
    print ("Statement {} mappings".format (self.name))
    for dd in self.map:
      print ("{}[{}]: {}".format (self.name, dd, self.map[dd]))
    for ref in self.accs:
      ref.show_maps ()

  # Draw tikz graph for statement using its mapping
  def print_tikz_graph (self, fout, par_x, par_y):
    for dd in self.map:
      mu = self.map[dd]
      nodename='{}_i{}'.format (self.name, dd)
      dimname = self.dims[dd]
      nodelabel = '{\\large\\textbf{ ' + '{}({})'.format (self.name, dimname) + '}}'
      if (mu < 0):
        nodelabel = '{\\large\\textbf{' + '{}({})=*'.format (self.name, dimname) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node [shape=rectangle,draw=blue,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      ## Print edges
      if (mu>=0):
        procdim = 'p{}'.format (mu)
        src = '({}.east)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[left] {} {};'.format ('->,line width=1mm,blue',src,'{}',tgt)
        fout.write (command + '\n')
    return len(self.map)

  # Draw tikz graph for statement using its mapping
  def print_ref_tikz_graph (self, fout, par_x, par_y):
    ref = self.accs[0]
    for dd in self.dims:
      dim_name = self.dims[dd]
      pi = ref.get_pi_by_dim_name_if_used (dim_name)
      if (pi == DIM_NOT_USED):
        continue
      nodename='{}_i{}'.format (ref.get_name (), dd)
      nodelabel = '{\\large\\textbf{ ' + '{}[{}]'.format (ref.get_name (), dim_name) + '}}'
      if (pi < 0):
        nodelabel = '{\\large\\textbf{' + '{}[{}]=*'.format (ref.get_name (), dim_name) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node[shape=rectangle,draw=red,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      if (pi >= 0):
        procdim = 'p{}'.format (pi)
        src = '({}.west)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[right] {} {};'.format ('->,line width=1mm,red',src,'{}',tgt)
        fout.write (command + '\n')
    return len(ref.get_dims ())

  ## @Statement: return True if the given dimension is mapped.
  def is_dimension_mapped (self, idim):
    if (self.map[idim] >= 0):
      return True
    return False

  # Return the processor map for the given iteration space dimension.
  def get_proc_dim_map(self, idim):
    return self.map[idim]

  # @Statement: Determine the lexico-positivity of the mu-mapping associated to the
  # current statement. This is used to later determine if the tiled layout
  # of an array has been changed, e.g. whether the layout [[A,B],[C,D]] morphed
  # into [[A,C],[B,D]] after an Allgather.
  def is_mu_lexico_positive (self, ref, producers = None):
    all_gat_follows = False
    if (producers != None and ref.get_name() in producers):
      all_gat_follows = producers[ref.get_name ()].was_allgathered (ref, self)
    temp = []
    NOMAP = 99
    for dd in range(self.ndim):
      pdim = self.map[dd]
      if (pdim >= 0):
        temp.append (pdim)
      else:
        temp.append (NOMAP)
    for dd in range(1,self.ndim):
      ## Must handle the degenerated case of some processor dimension
      ## having only 1 processor along it. E.g., 16x1.
      np_prev = 0;
      if (temp[dd-1] != NOMAP):
        np_prev = self.PP.get_dim_size (temp[dd-1])
      np_next = 0;
      if (temp[dd] != NOMAP):   
        np_next = self.PP.get_dim_size (temp[dd])
      ## Two cases:
      ## 1) Dimension is unmapped and next dimension is mapped and all gather follows
      ## 2) Back-to-back dimensions are mapped but the first one has size 1, and all gather follows
      if ((temp[dd-1] == NOMAP and temp[dd] != NOMAP and np_next > 1 and all_gat_follows) or
          (temp[dd-1] != NOMAP and temp[dd] != NOMAP and np_prev == 1 and np_next > 1 and all_gat_follows)):
        return False
    return True

  # @Statement: Operator must be the original producer of the array.
  # Operator must have been fully distributed.
  def was_allgathered (self, ref, stmt):
    if (len(self.accs) != 1):
      print ("[ERROR@was_allgathered]: was_allgathered() should only be used when the operator is the original producer of an array slice (1) - Operator is not a generator.")
      sys.exit (42)
    if (ref.get_name () != self.accs[0].get_name ()):
      print ("[ERROR@was_allgathered]: was_allgathered() should only be used when the operator is the original producer of an array slice (2) - Operator is *NOT* the producer of {}".format (ref.get_name ()))
      sys.exit (42)
    ## Allgather is necessary if the work is partitioned but the data
    ## must end up in a replicated fashion.
    ## This method is part of the generator, but it's being ultimately
    ## invoked from some other statement.
    for dd in range(self.get_num_dim()):
      idim_name = self.get_dim_name (dd)
      mu_dim = self.get_mu_dim_map (dd)
      ref_at_gen = self.accs[0]
      pi_dim = ref_at_gen.get_pi_by_dim_name_if_used (idim_name)
      proc_dim_size = 1
      if (mu_dim >= 0):
        proc_dim_size = self.PP.get_dim_size (mu_dim)
      if (option_debug >= 2):
        print ("Debug All-Gather from Statement {}.{}[{}], Generator={} - ref={} : pi={}, mu={}, proc-dim-size={}".format (stmt.get_name (), ref_at_gen.get_name (), idim_name, self.name, ref_at_gen.get_as_str(), pi_dim, mu_dim, proc_dim_size))
      if (pi_dim == DIM_UNMAPPED and mu_dim >= 0 and proc_dim_size > 1):
        return True
    return False

  ## Determine whether the layout of a matrix has changed. 
  ## For the layout to change, it must first be lexico-negative and it must
  ## have been all-gathered afterwards.
  def layout_changed (self, ref):
    sys.exit (9999)
    if (self.was_allgathered (ref)):
      return True
    if (self.is_mu_lexico_positive (ref)):
      return True
    return False
    

  # @Statement: Find the loop dimension corresponding to an iterator
  def get_dim_by_name (self, iter_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_name):
        return dd
    return -1

  def get_dim_name (self, idim):
    return self.dims[idim]

  def get_num_dim (self):
    return len(self.dims)

  # Return mapped processor dimension associated to an iteration space 
  # dimension identified by the name of the latter.
  def get_proc_map_by_dim_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return -1

  # Return a comma-separated list containing the number
  # of processors along each dimension. If the statement is
  # mapped at the current dimension, then we include '1',
  # otherwise we include the max number of dimensions.
  def get_processor_geometry_str_list (self, ref, PP):
    ret = ''
    for dd in self.map:
      iter_name = self.dims[dd]
      if (not ref.is_dim_used (iter_name)):
        continue
      if (not ret == ''):
        ret += ', '
      pdim = self.map[dd]
      if (pdim >= 0):
        ref_pdim = ref.get_proc_map_by_dim_name (iter_name)
        if (pdim == ref_pdim): 
          # If it's a match, and pdim >= 0, then access should be for a tile. 
          # Hence, we don't need the number of processor along the dimension pdim.
          ret += '1'
        elif (ref_pdim < 0): 
          # Statement still mapped. 
          # Local processor stores full extent of array dimension. 
          # Hence, will need the number of processors.
          ret += str(PP.get_dim_size (pdim))
        elif (ref_pdim >= 0 and ref_pdim != pdim):
          ret += str(PP.get_dim_size (pdim))
        else:
          ret += 'ERROR' # WEIRD CASE
      else:
        # Dimension is unmapped, so return the max among all the 
        # processor dimensions.
        ret += '1'
    return ret

  # Return statement expression representing the volume of a data tile
  def get_tile_vol (self, ref, ttc):
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (ref.is_dim_used (iter_name)):
        if (ret != ''):
          ret += ' * '
        extent = ref.get_extent_as_str_by_dim_name (iter_name)
        denum = '1'
        if (iter_name in ttc):
          denum = ttc[iter_name]
        expr = '{}(({}), {})'.format (DIMAGE_CEIL, extent, denum)
        if (int(denum) == 1):
          expr = extent
        ret += expr
    return ret

  def pretty_print_map (self, df):
    df.write ('<')
    for dd in self.map:
      if (dd > 0):
        df.write (', ')
      map_dim = self.map[dd]
      if (map_dim >= 0):
        df.write ('{}'.format (map_dim))
      else:
        df.write ('{}=*'.format (map_dim))
    df.write ('>')

  def get_name (self):
    return self.name

  ## Return the loop tripcount associated to the given dimension id.
  ## The argument dim_id must be between 0 and (depth-1).
  def get_loop_dim_tripcount (self, dim_id):
    dim_name = self.dims[dim_id]
    for ref in self.accs:
      if (ref.is_dim_used (dim_name)):
        return ref.get_dim_size_if_used (dim_name)
    return 0

  ## Statement.
  def estimate_total_iterations (self):
    res = 1
    for dd,dname in enumerate(self.dims):
      res = res * self.get_loop_dim_tripcount (dd)
    return res


  ## Statement.get_map_varname ():
  ## Return the mu (map) variable name. Not to confuse with
  ## method get_mu_varname defined within the Reference class.
  def get_map_varname (self, idim, pdim):
    varname = 'mu_{}_i{}_p{}'.format (self.name, idim, pdim)
    return varname

  def get_mu_sum_varname (self, dim_id):
    varname = 'sum_mu_{}_i{}_pX'.format (self.name, dim_id)
    return varname

  ## Statement.get_sum_reduction_mu_expr_along_dim ():
  ## Return the sum of mu variables that are used on a concrete processor
  ## dimension and that are also a reduction dimension.
  ## On matmul this amount to a sum of a single term, while for
  ## mttkrp this results in as many terms as reduction dimensions.
  def get_sum_reduction_mu_expr_along_dim (self, ref, pdim):
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (self.is_reduction_dim (ref, iter_name)):
        if (ret != ''):
          ret += ' + '
        ret += self.get_map_varname (dd, pdim)
    return ret

  def is_write_ref (self, ref):
    if (self.is_data_sink ()):
      return False
    nref = len(self.accs)
    write_ref = self.accs[nref-1]
    return (write_ref.get_name () == ref.get_name ())
      
    
  def writeln(self, mf, line):
    mf.write(line + "\n")

  def add_constraint (self, mf, cstr, comment = ''):
    self.writeln (mf, 'opt.add ({}) ## {}'.format (cstr, comment))
    self.cof.add_cstr (cstr)

  def set_lower_bound (self, mf, varname, lb):
    plain_cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_upper_bound (self, mf, varname, ub):
    plain_cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_bounds (self, mf, varname, lb, ub):
    plain_cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  ## Operator.
  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    self.set_bounds (mf, varname, lb, ub)

  ## Statement.declare_variable 
  def declare_variable (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_boolean (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Bool('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_float (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Real('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_map_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Error. Dictionary is None.")
      sys.exit (42)
    NP = self.np
    for dd in self.dims:
      for pp in range(NP):
      #print ("Dim {}: {}".format (dd, self.dims[dd]))
        varname = self.get_map_varname (dd,pp)
        decl = self.declare_variable (mf, varname, decl)
        self.set_bounds_boolean (mf, varname)
    return decl

  ## Statement.
  def set_sum_bound_along_dim (self, mf, mode, dim, ub, decl):
    nn = self.ndim
    if (mode == PER_DIM):
      nn = self.np
    cstr = ""
    for kk in range(nn):
      if (not cstr == ""):
        cstr += " + "
      varname = ""
      if (mode == PER_DIM):
        varname = self.get_map_varname (dim, kk)
      if (mode == PER_PROC):
        varname = self.get_map_varname (kk, dim)
      cstr += varname
    cstr += " <= {}".format (ub)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  def set_dim_sum_bounds (self, mf, decl):
    for dd in range(self.ndim):
      self.writeln (mf, '## set_dim_sum_bounds')
      decl = self.set_sum_bound_along_dim (mf, PER_DIM, dd, 1, decl)
    return decl

  def set_proc_sum_bounds (self, mf, decl):
    for dd in range(self.np):
      self.writeln (mf, '## set_proc_sum_bounds; np={}'.format (self.np))
      decl = self.set_sum_bound_along_dim (mf, PER_PROC, dd, 1, decl)
    return decl

  ## Statement.
  ## Iterate through accessed tensors, and
  ## set sum bounds for each DS-dimension and each PS-dimension affecting
  ## the pi-mappings of the current reference.
  def set_ref_sum_bounds (self, mf, decl, LT = None):
    for rr in self.refs:
      ref = self.refs[rr]
      if (LT != None and not rr in LT):
        decl = ref.set_dim_sum_bounds (mf, decl)
        decl = ref.set_proc_sum_bounds (mf, decl)
    return decl

  ## Statement.
  ## Declare pi-mapping variables for the current statement.
  ## FFT-modification: Added parameter LT to skip declarations and 
  ## constraints on linearized tensors.
  def declare_ref_vars (self, mf, decl, LT = None):
    if (decl == None):
      print ("[ERROR] Error in dictionary.")
      sys.exit (42)
    for ref in self.accs:
      if (LT != None and not ref in LT):
        decl = ref.declare_map_vars (mf, decl)
    return decl
    
  ## Link mu and pi variables
  def link_dimensions (self, mf):
    for pp in range(self.np):
      for dd in self.dims:
        dim = self.dims[dd]
        for rr in self.refs:
          ref = self.refs[rr]
          muvar = self.get_map_varname (dd, pp)
          ref.link_dimensions (mf, pp, dd, dim, muvar)

  def get_comm_slice_variable (self, ref_name, dim_id):
    varname = 'K_{}_{}_{}'.format (self.name, ref_name, dim_id)
    return varname

  def get_comm_ref_variable (self, ref_name):
    varname = 'K_{}_{}'.format (self.name, ref_name)
    return varname

  def declare_comm_slice_variable (self, mf, ref, dim_id, decl):
    varname = ref.get_local_ref_vol_var (self.name)
    decl = self.declare_variable (mf, varname, decl)
    return decl

  def declare_comm_ref_variable (self, mf, ref, decl):
    varname = ref.get_local_ref_vol_var (self.name)
    decl = self.declare_variable (mf, varname, decl)
    return decl

  def set_comm_slice_function (self, mf, ref_name, dim_id, slice_var, pbs):
    cstr_sum = ""
    cstr_prod = ""
    cstr = ''
    USE_OLD = False
    # The loop below creates the expression: N / (sum Pi x pi_var  + prod (1 - pi))
    # Which is expanded and simplified into:
    # sum ( pi_var x N/Pi ) + N * (1 - sum pi)
    # The above works because sum of pi variables is guaranteed to be upper bounded by 1.
    if (USE_OLD):
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dim_id, pp)
        term1 = '{} * {}'.format (proc_var, mu_var)
        term2 = '(1 - {})'.format (mu_var)
        if (pp > 0):
          cstr_sum += " + "
          cstr_prod += " * "
        cstr_sum += term1
        cstr_prod += term2
      self.writeln (mf, '## Defined in set_comm_slice_function')
      cstr = '{} == {} / ({} + {})'.format (slice_var, pbs, cstr_sum, cstr_prod)
    else:
      portions = []
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dim_id, pp)
        if (pp > 0):
          cstr_sum += " + "
          cstr_prod += " + "
        Nportion = ''
        if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          Nportion = int(math.ceil(int(pbs) * 1.0/int(proc_var)))
        else:
          Nportion = '({} / {})'.format (pbs, proc_var)
        portions.append (Nportion)
        cstr_sum += '{} * {}'.format (Nportion, mu_var)
        cstr_prod += mu_var
        # lower bound constraints for slice_var
        if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          cstr_lb = '{} >= {} * {}'.format (slice_var, Nportion, mu_var)
          cmd = 'opt.add ({}) # parametric lower bound'.format (cstr_lb)
          self.add_constraint (mf, cstr_lb, ' cap. lower bound of communicated data')
      cstr = '{} == {} + {} - {} * ({})'.format (slice_var, cstr_sum, pbs, pbs, cstr_prod)
    self.cof.add_cstr (cstr)
    cmd = 'opt.add ({})'.format (cstr) 
    self.writeln (mf, cmd)
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      cstr = '{} >= {}'.format (slice_var, min(portions))
      cmd = 'opt.add ({})'.format (cstr) 
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    if (USE_MODULO):
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        cstr = '{} % {} == 0'.format (pbs, proc_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  ## Statement.
  ## Minor modification to support FFT.
  ## Declare and define a variable "Local_<stmt>_<ref>".
  def define_comm_slice (self, mf, ref, dim_id, decl):
    ref_name = ref.get_name ()
    #slice_var = self.get_comm_slice_variable (ref_name, dim_id)
    #slice_var = ref.get_local_ref_vol_var (self.name)
    slice_var = ref.get_local_ref_dim_vol_var (stmt.name, dim_id)
    decl = self.declare_variable (mf, slice_var, decl)
    extent = ref.get_array_extent_by_dim_name (self.dims[dim_id])
    decl = self.declare_comm_slice_variable (mf, ref, dim_id, decl)
    self.set_upper_bound (mf, slice_var, extent)
    self.set_comm_slice_function (mf, ref_name, dim_id, slice_var, extent)
    return decl

  ## Create the constraint on variables Local_{<stmt>,<ref>}
  ## and on the corresponding dimension variables, Local_{<stmt>,<ref>,<dim>}.
  ## Also add lower bounds: the Local_{stmt,ref} >= Local_{stmt,ref,dim>
  def set_comm_slice_expressions (self, mf, decl, LT, pnc):
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      decl = self.declare_comm_ref_variable (mf, ref, decl)
      #decl = ref.get_local_ref_vol_var (self.name)
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      local_comm = ref.get_local_ref_vol_var (self.name)
      ref_name = ref.get_name ()
      comm_var = self.get_comm_ref_variable (ref_name)
      expr = ""
      for dd in self.dims:
        dim_name = self.dims[dd]
        if (ref.is_dim_used (dim_name)):
          decl = self.define_comm_slice (mf, ref, dd, decl)
          if (not expr == ""):
            expr += " * "
          local_comm_var = ref.get_local_ref_dim_vol_var (stmt.name, dd)
          expr += local_comm_var
          cstr = '{} >= {}'.format (local_comm, local_comm_var)
          self.add_constraint (mf, cstr)
      # Alternate between '>=' and '=='
      # Prefer '>=' over '=='. We are computing an upper bound after all.
      # Individual slice contributions from each array will be exact.
      cmd = '{} == {}'.format (local_comm, expr) #comm_var, expr)
      #self.writeln (mf, cmd)
      self.add_constraint (mf, cmd, ' Local accessed volume of tensor at operator.')
      ## Set at upper-bound for the local volume.
      cstr = '{} <= {}'.format (local_comm, pnc)
      self.add_constraint (mf, cstr, ' Upper bound of locally accessed volume at operator.')
    return decl

  ## Statement.
  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname
     

  ## Create capacity constraints per statement.
  ## The maximum capacity is given for the whole program.
  ## The memory needed by a statement results from the sum of all its parts.
  def set_statement_capacity_constraint (self, mf, decl, pnc, maxprocs):
    total_expr = ''
    self.writeln (mf, "## Introduced by stmt.set_statement_capacity_constraint")
    total_var = self.get_volume_var ()
    decl = self.declare_variable (mf, total_var, decl)
    nn = len(self.accs)
    rid = 1
    for ref in self.accs:
      if (not total_expr == ''):
        total_expr += " + "
      # Use the current volume var as a lower bound of the total volume
      volvar = ref.get_volume_var ()
      is_comp_op = True
      decl = ref.define_volume_var (mf, decl, is_comp_op)
      # Set an 'easy' lower-bound for the req_{stmt} variables
      cstr = '{} * {} >= {}'.format (maxprocs, total_var, volvar)
      cmd = 'opt.add ({}) # See set_statement_capacity_constraint ()'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
      ## Compensate for potential 3-buffers and MPI internal storage.
      ## Extra space for buffers.
      if (rid < nn):
        total_expr += volvar
      else:
        total_expr += '{} * {}'.format (DIMAGE_CAP_FACTOR, volvar)
      rid += 1
    # Only insert the equality: req_ss = \sum_{aa} req_{ss,aa} if we have
    # two or more arrays used ss.
    if (len(self.accs) > 1):
      cstr = '{} == {}'.format (total_var, total_expr)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    if (pnc > 0):
      cstr = '{} <= {}'.format (total_var, pnc)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    ## Add the default per-node capacity (pnc) upper-bound. (May 2024)
    return decl

  def get_comm_var (self):
    varname = 'K_{}'.format (self.name)
    return varname

  # Return the name of a computation cost variable (W / omega)
  def get_comp_cost_variable (self):
    varname = 'W_{}'.format (self.name)
    return varname

  ## Return the Global Performance Objective variable name of the
  ## current statement.
  def get_gpo_varname (self):
    varname = 'G_{}'.format (self.name)
    return varname

  def get_sanity_varname (self):
    varname = 'Z_{}'.format (self.name)
    return varname

  def get_ratio_varname (self):
    varname = 'R_{}'.format (self.name)
    return varname

  ## To avoid spurious mappings and matchings.
  def get_parity_check_expression (self, ref):
    ret = ''
    for idim in self.dims:
      dim_name = self.dims[idim]
      if (ref.is_dim_used (dim_name)):
        term = ''
        adim = ref.get_dim_if_used (dim_name)
        if (adim < 0):
          continue
        sum_pi_var = ref.get_sum_pi_var_along_dim (adim, -1)
        for pdim in range(self.PP.get_num_dim ()):
          pi_var = ref.get_pi_var_by_dim_name (dim_name, pdim)
          mu_var = self.get_map_varname (idim, pdim)
          if (term != ''):
            term += ' + '
          term += '(({}+{})%2)'.format (mu_var, pi_var)
        if (ret != ''):
          ret += ' + '
        term = '({}) * {}'.format (term, sum_pi_var)
        ret += term
    return '(' + ret + ') * {}'.format (1)   #MEM2COMP_RATIO)


  ## Statement.set_comm_constraints():
  ## Introduce for each statement (operator), communication volume
  ## constraints tying lambdas (matching variables) with effective volumes,
  ## L^{S,A}. Outgoing arrays also use rhos (replication) variables.
  ## Build the communication constraint for a statement.
  ## K_ss = \sum_ref K_{ss,ref}
  ## We skip read constraints for generators and write communications 
  ## constraints for data sinks.
  ## Further, we also add constraints of the form
  ## \forall ref: K_ss >= K_{ss,ref}
  ## K-constraints
  def set_comm_constraints (self, mf, decl, LT):
    ## Review this
    ## Modification for FFT support.
    temp = []
    for ref in self.accs:
      temp.append (ref)
    last = len(self.accs)
    temp.append (self.accs[last-1])
    total_expr = ''
    total_var = self.get_comm_var ()
    decl = self.declare_variable (mf, total_var, decl)
    ## In the future, it might be of interest to fine-tune between 
    ## equalities and inequalities.
    OPERATOR = '=='
    rep_factor = ''
    for ii,ref in enumerate(temp):
      if (self.is_data_sink () and ii > 0):
        continue
      if (self.is_data_generator () and ii < last):
        continue
      if (ref.get_name () in LT):
        continue
      if (not total_expr == ''):
        total_expr += " + "
      umv = ref.get_match_variable (self.name)
      commvar = ref.get_stmt_read_ref_comm_var (self.name) # commvar is L^{S,A}
      rho_var = ref.get_rho_varname ()
      if (ii == last):
        commvar = ref.get_stmt_write_ref_comm_var (self.name) 
      volvar = ref.get_volume_var ()
      decl = ref.define_stmt_ref_local_vol_var (mf, self.name, decl)
      local_comm = ref.get_local_ref_vol_var (self.name)
      local = ref.get_match_variable (self.name)
      penalty = ''
      if (DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS):
        penalty = '{} == 0'.format (self.get_parity_check_expression (ref))        
        self.add_constraint (mf, penalty)
        penalty = ''
      else:
        # Will allow cross-dimension mappings as a penalty to the solution. Will 
        # increase time-to-solution.
        penalty = ' + {}'.format (self.get_parity_check_expression (ref))
      decl = self.declare_variable (mf, commvar, decl)
      ## Alternate between '==' and '>='. Will use variable OPERATOR defined above.
      term = '{} {} {} * (1 - {}{})'.format (commvar, OPERATOR, local_comm, umv, penalty)
      # Use the current commvar as the lower bound of the totalvar
      cstr = '{} >= {}'.format (total_var, commvar)
      self.add_constraint (mf, cstr, ' setting lower bound of total var')
      if (ii == last): 
        ## This equality means the reference is a write.
        ## Eventually, may alternate between '==' and '>='
        ## Changed the '(rho_var)' to '(1 - rho_var)'. 
        ## Replication translates to all-reduce.
        decl = ref.set_rho_var_dim_constraints (mf, decl, self)
        rep_factor = ref.get_replication_out_factor_expr (self.name)
        term = '{} == {} * ({})'.format (commvar, local_comm, rep_factor)
      self.add_constraint (mf, term)
      ## Add 0 as lower bound constraint.
      cstr = '{} >= 0'.format (commvar)
      self.add_constraint (mf, cstr)
      ## Add upper bound constraint for commvar.
      cstr = '{} <= {}'.format (commvar, local_comm)
      self.add_constraint (mf, cstr)
      total_expr += commvar
    # Set: comm^s >= sum comm^{s,ref}
    if (total_expr != ''):
      cstr = '{} {} {}'.format (total_var, OPERATOR, total_expr)
      cmd = 'opt.add ({}) # set_comm_constraints'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    return decl

  def get_objective_name (self, otype):
    varname = 'obj_{}_{}'.format (otype, self.name)
    return varname

  def set_req_objective (self, mf, max_obj):
    varname = self.get_objective_name ('req')
    obj_mode = ''
    if (max_obj):
      obj_mode = 'maximize'
    if (not max_obj):
      obj_mode = 'minimize'
    obj_var = self.get_volume_var ()
    cmd = '{} = opt.{}({})'.format (varname, obj_mode, obj_var)
    self.writeln (mf, cmd)

  def set_comm_objective (self, mf, max_obj):
    varname = self.get_objective_name ('K')
    obj_mode = ''
    if (max_obj):
      obj_mode = 'maximize'
    if (not max_obj):
      obj_mode = 'minimize'
    obj_var = self.get_comm_var ()
    cmd = '{} = opt.{}({})'.format (varname, obj_mode, obj_var)
    self.writeln (mf, cmd)

  ## Add the constraints of the form:
  ## LM_ss in [0,1]
  ## LM_{ss,aa,idim} = (1 - \sum_{p} pi_{aa,adim,p} + \sum_{pp} pi_{aa,adim=idim,pp} x pi_{ss,adim=idim,pp}
  ## LM variables correspond to the \lambda variables used in the paper.
  def add_matching_constraints (self, mf, decl, LT):
    ## Local match modifications.
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      ref_match_var = ref.get_match_variable (self.name)
      decl = self.declare_variable (mf, ref_match_var, decl)
      self.set_bounds_boolean (mf, ref_match_var)
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      for dd in self.dims:
        dim_name = self.dims[dd]
        if (ref.is_dim_used (dim_name)):
          sum_mu_var = self.get_mu_sum_varname (dd)
          decl = ref.declare_matching_variables (mf, self.name, dd, sum_mu_var, dim_name, decl)
        else:
          print ("WARNING: dimension {} not used in reference {}".format (dim_name, ref.get_name ()))
    # local_match = (1 - sum pi) + sum pi_d x mu_d
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      ref_match_var = ref.get_match_variable (self.name)
      USE_PROD = False
      USE_PROD = True
      if (USE_PROD):
        match_expr = ""
        for dd in self.dims:
          dim_name = self.dims[dd]
          if (ref.is_dim_used_in_linear_group (dim_name)): 
            if (not match_expr == ""):
              match_expr += " * "
            ref_dim = ref.get_dim_if_used (dim_name)
            ref_dim = ref.get_dim_if_used_in_linear_group (dim_name)
            ref_match_dim_var = ref.get_match_dim_variable (self.name, dd) #
            match_expr += ref_match_dim_var 
        cstr = '{} == {}'.format (ref_match_var, match_expr)
        self.add_constraint (mf, cstr)
      else:
        for dd in self.dims:
          dim_name = self.dims[dd]
          if (ref.is_dim_used (dim_name)):
            ref_match_dim_var = ref.get_match_dim_variable (self.name, dd)
            cstr = '{} == {}'.format (ref_match_var, ref_match_dim_var) 
            self.add_constraint (mf, cstr)
    return decl

  ## Stmt
  def declare_replication_variables (self, mf, decl, LT):
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      decl = ref.declare_replication_variables (mf, decl, LT)
    return decl

  def bound_replication_variables (self, mf, LT):
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      ref.bound_replication_variables (mf)

  def add_replication_constraints (self, mf, LT):
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      ref.link_rho_variables (mf)

  def set_array_dim_replication_expression (self, mf, LT):
    for ref in self.accs:
      if (ref.get_name () in LT):
        continue
      ref.link_replication_to_placement (mf)

  def declare_block_variables (self, mf, decl):
    for ref in self.accs:
      decl = ref.declare_block_variables (mf, decl)
    return decl


  ## Create a sum variable for each mu variable, i.e.:
  ## \forall i: sum_mu_i = \sum_{j} \mu_{i,j}
  def set_mu_dimension_sum (self, mf, decl):
    for dd in self.dims:
      mu_sum_var = self.get_mu_sum_varname (dd)
      decl = self.declare_variable (mf, mu_sum_var, decl)
      sum_expr = ''
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dd, pp)
        if (sum_expr != ''):
          sum_expr += ' + '
        sum_expr += mu_var
      expr = '{} == {}'.format (mu_sum_var, sum_expr)
      cmd = 'opt.add ({})\n'.format (expr)
      self.set_bounds_boolean (mf, mu_sum_var)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    return decl

  ## Build the computation cost expression of the form:
  ## \forall i: (\sum_j N_i * \mu_{i,j} / P_j) - N_i * mu_sum_i + N_i
  ## where i is a loop dimension, j is a processor dimension
  ## and mu_sum_i = \sum_j \mu_{i,j}.
  def set_computation_cost_expression (self, mf, decl):
    varname = self.get_comp_cost_variable ()
    decl = self.declare_variable (mf, varname, decl)
    cost_expr = ''
    all_min = []
    for dd in self.dims:
      expr = ''
      tripcount = self.get_loop_dim_tripcount (dd)
      mu_sum_var = self.get_mu_sum_varname (dd)
      size_list = []
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dd, pp)
        Nportion = ''
        term = ''
        if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          Nportion = int(math.ceil (int(tripcount) * 1.0 / proc_var))
          size_list.append (Nportion)
        else:
          Nportion = '({} / {})'.format (tripcount, proc_var)
        term = '({} * {})'.format (Nportion, mu_var)
        if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          ## Lower bound of dimension
          cstr_lb = '{} >= {} * {}'.format (varname, Nportion, mu_var)
          cmd = 'opt.add ({}) # lb w-check'.format (cstr_lb)
          self.writeln (mf, cmd)
          self.cof.add_cstr (cstr_lb)
        if (expr != ''):
          expr += ' + '
        expr += term
      if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
        all_min.append (min(size_list))
      no_map_term = '{} * (1 - {})'.format (tripcount, mu_sum_var)
      factor = '({}) + {}'.format (expr, no_map_term)
      if (cost_expr != ''):
        cost_expr += ' * '
      cost_expr += '({})'.format (factor)
    # Set a better lower bound for the var
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      expr = '{} >= {}'.format (varname, prod(all_min))
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    cost_expr = '{} == {}'.format (varname, cost_expr)
    cmd = 'opt.add ({})'.format (cost_expr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cost_expr)
    ## Upper bound of dimension
    USE_PER_DIM_UB = True
    USE_MAXLOCALVOL_UB = not USE_PER_DIM_UB
    if (USE_PER_DIM_UB):
      #### NOTE: Upper bound for work of matmul-like operator must be total number of local iterations of the original serial operator.
      for pp in range(self.PP.get_num_dim ()):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        niters = self.estimate_total_iterations ()
        cstr_ub = '{} <= ({}) / ({})'.format (varname, niters, proc_var)
        cmd = 'opt.add ({}) # ub w-check'.format (cstr_ub)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr_ub)
    elif (USE_MAXLOCALVOL_UB):  # Unsafe for small problems, but fast.
      op_ub = self.compute_max_local_volume () * MEM2COMP_RATIO 
      cstr_ub = '{} <= {}'.format (varname, op_ub)
      cmd = 'opt.add ({}) # ub w-check'.format (cstr_ub)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr_ub)
    else:
      cstr_ub = '{} <= {}'.format (varname, self.estimate_total_iterations ())
      cmd = 'opt.add ({}) # ub w-check'.format (cstr_ub)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr_ub)
    return decl

  ## Operator
  ## 
  def compute_max_local_volume (self):
    ret = 0
    for ref in self.accs:
      ret += ref.get_full_tensor_volume ()
    return ret

  ## Build a performance expression constraint for the current statement.
  ## The expression will be of the form: work_cost + aplha * comm_cost,
  ## where alpha is defined as a machine specific memory-to-compute ratio.
  ## For each compute-statement ss do:
  ## G_ss = K_ss + 40 x W_ss
  ## where G is the global cost, K is the communication cost and W is the 
  ## computation cost.
  def set_performance_expression_constraints (self, mf, decl, obj_mode):
    objvar = self.get_gpo_varname () ## gov = Global Performance Objective
    decl = self.declare_variable (mf, objvar, decl)
    k_comm_var = self.get_comm_var ()
    w_comp_var = self.get_comp_cost_variable ()
    ## Alternate, eventually between == or >=.
    ## Default COMM_ONLY objective mode
    expr = '{} >= {}'.format (objvar, k_comm_var)
    ## Below: performance objective for current statement.
    if (obj_mode == DIMAGE_OBJ_COMM_COMP):
      expr = '{} == {} + {} * {}'.format (objvar, w_comp_var, MEM2COMP_RATIO, k_comm_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
      ## Upper-bound for operator.
      max_single_node_vol = self.compute_max_local_volume ()
      expr = '{} <= {}'.format (objvar, MEM2COMP_RATIO * max_single_node_vol * self.PP.get_max_procs () * int(self.PP.get_num_dim ()))
      scale_factor = 1
      if (self.is_compute_statement ()):
        scale_factor = self.PP.get_proc_sum_expr ()
      expr = '{} <= {} * ({})'.format (objvar, MEM2COMP_RATIO * max_single_node_vol, scale_factor)  ## May 21
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)

    # Set lower bounds for performance for each component.
    if (obj_mode == DIMAGE_OBJ_COMM_COMP):
      ## Lower bound G with W
      expr = '{} >= {}'.format (objvar, w_comp_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
      ## Lower bound G with MEM2COMP_RATIO x K
      expr = '{} >= {} * {}'.format (objvar, 1, k_comm_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    else:
      ## Lower bound G with MEM2COMP_RATIO x K
      expr = '{} >= {}'.format (objvar, k_comm_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    return decl

  ## Statement (linearizer: Tensor to Matrix conversion cost).
  ## Will be a linearizer, not a DFT operator.
  def set_linearizer_communication_constraint (self, mf, decl, LT):
    if (len(self.accs) != 2):
      print ('[ERROR] Linearizer operator must access exactly two references.')
      sys.exit (42)
    tensor = LT[self.accs[0].get_name ()]
    matrix = self.accs[1]
    if (tensor.ndim < matrix.ndim):
      tensor = LT[self.accs[1].get_name ()]
      matrix = self.accs[0]
    gpo_var = self.get_gpo_varname () ## gov = Global Performance Objective
    decl = self.declare_variable (mf, gpo_var, decl)
    decl = tensor.set_linearizer_communication_constraint (mf, decl, matrix, self.dims, None)
    lcvar_main = tensor.linearizer_cost_var (matrix)
    cstr = '{} == {}'.format (gpo_var, lcvar_main)
    self.add_constraint (mf, cstr)
    return decl

  def set_sanity_check_constraints (self, mf, decl):
    k_comm_var = self.get_comm_var ()
    w_comp_var = self.get_comp_cost_variable ()
    z_var = self.get_sanity_varname ()
    ratio_var = self.get_ratio_varname ()
    decl = self.declare_boolean (mf, z_var, decl)
    expr = '{} == ({} / {} <= {})'.format (z_var, k_comm_var, w_comp_var, MEM2COMP_RATIO * 2)
    self.writeln (mf, expr)
    self.cof.add_cstr (expr)
    return decl


  def operator_extract_dims_from_mu_var (self, mu_var):
    parts = mu_var.split ("_")
    print ("Operator {} - parts = ##{}##".format (self.name, parts))
    idim_str = re.sub ("i","",parts[2])
    idim = int(idim_str)
    pdim_str = re.sub ("p","",parts[3])
    pdim = int(pdim_str)
    return (idim,pdim)

  ## Statement.extract_mappings_from_solution_set(): 
  ## Extract the values of the mu variables from the solution set.
  def extract_mappings_from_solution_set (self, solset):
    for vv in solset:
      if (vv.find ("sum") >= 0):
        continue
      muprefix='mu_{}_'.format (self.name)
      if (vv.find (muprefix) == 0):
        if (int(solset[vv]) == 1):
          idim, pdim = self.operator_extract_dims_from_mu_var (vv)
          self.map[idim] = pdim
    for ref in self.accs:
      ref.extract_mappings_from_solution_set (solset)

  ## Statement:
  ## Traverse the list of accesses and add the array names to the
  ## @arrset dictionary. Return the updated dictionary.
  def collect_arrays (self, arrset):
    for ref in self.accs:
      if (ref.get_name () in arrset):
        continue
      arrset[ref.get_name ()] = ref
    return arrset

  def collect_communicators (self, comms):
    for ref in self.accs:
      comms = ref.collect_communicators_for_statement (self.name, comms)
    return comms

  # Produce a vector of iteration space dimensions which appear
  # in the @ref reference argument. The following holds for each
  # entry v_i in the vector:
  # v_i == -1: dimension i is not used in ref
  # v_i >= 0: dimension i is used in ref[v_i]
  # The vector is finalized with a -2.
  def generate_ref_udim_declarations (self, mf, ref):
    dimlist = ""
    for dd in self.dims:
      dim_name = self.dims[dd];
      entry = ref.get_dim_if_used (dim_name)
      dimlist += '{}'.format (entry)
      dimlist += ', '
    dimlist += '-2'
    varname = ref.get_udim_varname (self.name)
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def generate_udim_declarations (self, mf):
    for ref in self.accs:   
      self.generate_ref_udim_declarations (mf, ref)

  def get_imap_varname (self):
    varname = 'DIMAGE_IMAP_{}'.format (self.name)
    return varname

  def generate_stmt_imap_declarations (self, mf):
    dimlist = ""
    for dd in self.map:
      dimlist += '{}'.format (self.map[dd])
      dimlist += ', '
    dimlist += '-2'
    varname = self.get_imap_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def generate_communicators (self, df):
    for ref in self.accs:
      ref.generate_communicators_at_statement (df, self)
      df.write ('\n')
    
  def declare_communicators (self, df):
    for ref in self.accs:
      ref.declare_communicator_at_statement (df, self)
  
  def get_operator_name (self):
    opname = 'dimage_operator_{}'.format (self.name)
    return opname

  def indent (self, df):
    df.write ('  ')

  ## Determine if the current statement is a data generator.
  ## Assume that this is obtained from the statement's name, which
  ## must be prefixed with the keyword 'gen'.
  def is_data_generator (self):
    if (self.name.find ('gen') == 0):
      return True
    if (self.name.find ('Gen') == 0):
      return True
    return False

  ## Determine if the current statement is a data sink.
  ## Assume that this is obtained from the statement's name, which
  ## must be prefixed with the keyword 'sink'.
  def is_data_sink (self):
    if (self.name.find ('sink') == 0):
      return True
    if (self.name.find ('Sink') == 0):
      return True
    return False

  def is_compute_statement (self):
    if (self.is_data_generator ()):
      return False
    if (self.is_data_sink ()):
      return False
    return True

  def get_iterator_variable (self, idim, is_point):
    varname = 'i{}'.format (idim)
    if (not is_point):
      varname = 'b{}'.format (idim)
    return varname

  def is_input_array (self, ref):
    if (self.is_data_generator ()):
      return False
    if (self.is_data_sink ()):
      return self.accs[0].get_name () == ref.get_name ()
    if (self.accs[0].get_name () == ref.get_name ()):
      return True
    if (self.accs[1].get_name () == ref.get_name ()):
      return True
    return False

  def is_output_array (self, ref):
    if (self.is_data_sink ()):
      return False
    n_acc = len(self.accs)
    return self.accs[n_acc-1].get_name () == ref.get_name ()

  ## Statement.is_reduction_dim ():
  ## Return True if the given reference is a write-reference and if the
  ## given iterator doesn't appear in it. Return False otherwise.
  def is_reduction_dim (self, ref, iter_name):
    if (not self.is_output_array (ref)):
      return False
    return (not ref.is_dim_used (iter_name))

  ## @Statement: Determine whether the reduction dimension is being mapped
  ## or not. If it's in fact mapped, then we will need an allgather or allreduce
  def get_mapped_reduction_dimension (self, ref, PP):
    for dd in self.dims:
      iter_name = self.dims[dd]
      true_reduction = self.is_reduction_dim (ref, iter_name) and self.map[dd] >= 0
      space_reduction = not self.is_reduction_dim (ref, iter_name) and self.map[dd] >= 0 and ref.get_pi_by_name (iter_name) == -1
      if (true_reduction): # 
        pdim = self.map[dd]
        if (space_reduction):
          pdim = ref.get_pi_by_name (iter_name)
        if (option_debug >= 4):
          print ("\t\tOperator {} - Is reduction dim ({}:{}) : map[{}]={} - psize={}".format (self.name, dd, iter_name, dd, pdim, PP.get_dim_size (pdim)))
        if (PP.get_dim_size (pdim) > 1): #
          return dd
    return -1
    
  def has_mapped_reduction_dimension (self, ref, PP):
    has = self.get_mapped_reduction_dimension (ref, PP) >= 0
    return has

  def get_point_trip_count (self, idim, PP, extent, ttc):
    num_proc = '1'
    iter_name = self.dims[idim]
    if (iter_name in ttc):
      num_proc = ttc[iter_name]
    expr = '{}({}, {})'.format (DIMAGE_CEIL, extent, num_proc)
    return expr

  ## Statement: Construct the loop structure of the current statement.
  def build_loop_structure (self, idim, PP, is_point, producers, skip_red_dim, for_accum, only_ub = False):
    ret = '/* Loop generated externally */'
    return ret

  ## @Statement: return the number of blocks to be executed at a 'b' loop.
  def get_block_stride (self, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    ret = 1
    mu_dim = self.map[idim]
    if (iter_name in ttc and mu_dim >= 0):
      trip = ttc[iter_name]
      ret = self.PP.lcm () / int(trip)
    if (ret == 1 and mu_dim >= 0):
      return ret
    num_proc = 1
    if (mu_dim >= 0):
      num_proc = self.PP.get_dim_size (mu_dim)
    ret = ret / num_proc
    return ret

  ## Return the number of blocks that must be traversed for a given loop dimension
  ## and on a given array (ref).
  def get_block_stride_from_ref (self, ref, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    ret = 1
    if (iter_name in ttc):
      trip = ttc[iter_name]
      ret = self.PP.lcm () / int(trip)
    if (ret == 1):
      return ret
    num_proc = 1
    pi_dim = ref.get_proc_map_by_dim_name (iter_name)
    if (pi_dim >= 0):
      num_proc = self.PP.get_dim_size (pi_dim)
    ret = ret / num_proc
    return ret

  ## Return the number of blocks that must be traversed for a given loop dimension
  ## and on a given array (ref).
  def get_number_tiles_along_dim (self, ref, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    mu_dim = self.map[idim]
    pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
    ret = 1
    if (mu_dim == pi_dim  and mu_dim >= 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (pi_dim)
    elif (mu_dim >= 0 and pi_dim < 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (mu_dim)
    elif (pi_dim < 0):
      ret = self.PP.lcm ()
    elif (mu_dim < 0 and pi_dim >= 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (pi_dim)
    else:
      ret = 9999999999
    return ret

  ## @Statement
  def build_l2_loop (self, level, trip, skip_red_dim, gen_mode, producers = None):
    nref = len(self.accs)
    out_ref = self.accs[nref-1]
    iter_name = self.dims[level]
    red_dim_found = self.is_reduction_dim (out_ref, iter_name)
    if (skip_red_dim and option_debug >= 3):
      print ("[INFO] Showing reduction info>> found {}, expected {}".format (red_dim_found, level))
    if (skip_red_dim and red_dim_found):
      return '' #iter = {},  srd = {} - rdf = {}'.format (iter_name, skip_red_dim,red_dim_found)
    ref = None
    for rr in range(len(self.accs)):
      temp = self.accs[rr]
      if (temp.is_dim_used (iter_name)):
        ref = temp
        break
    bi='b{}'.format (level)
    ti='t{}'.format (level)
    if (trip == None):
      lcm = self.PP.lcm ()
      tiles = 1
      mu_dim = self.map[level]
      if (mu_dim >= 0):
        trip = self.PP.get_dim_size (mu_dim)
    if (trip == None):
      trip = 1
    stride = self.get_number_tiles_along_dim (ref, level, producers)
    l2_loop_lb = '{}*{}'.format (bi,stride)
    l2_loop_ub = '({}+1) * {} - 1'.format (bi,stride)
    if (gen_mode == L2_LOOP_GENMODE_LB):
      return l2_loop_lb
    if (gen_mode == L2_LOOP_GENMODE_UB):
      return l2_loop_ub
    self.ntpd.append (int(stride))
    ret = ''
    ret += 'for ({} = {}; /* lcm={} */'.format (ti, l2_loop_lb, self.PP.lcm ())
    ret += '{} <= {}; '.format (ti,l2_loop_ub)
    ret += '{}++)'.format (ti)
    return ret

  def build_cannonical_loop (self, level):
    iter_name = 'i{}'.format(level)
    trip_count = self.get_loop_dim_tripcount (level)
    indent = '  ' * (level + 1)
    ret = '{}for ({} = 0; {} < {}; {}++)'.format (indent, iter_name, iter_name, trip_count, iter_name)
    return ret


  # Return the list i0, i1, ... as a string and separated by commas.
  def get_iterator_str_list (self):
    ret = ''
    for ii,dd in enumerate(self.dims):
      if (ii > 0):
        ret += ', '
      ret += 'i{}'.format (ii)
    return ret


  # Return the list i0, i1, ... as a string and separated by commas,
  # and which are used in @ref.
  def get_iterator_str_list_used_in_ref (self, ref):
    ret = ''
    used = 0
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      if (idim >= 0):
        if (used > 0):
          ret += ', '
        ret += 'i{}'.format (idim)
        used += 1
    return ret

  def get_sliced_iterator_str_list_used_in_ref (self, ref):
    ret = ''
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      mu_dim = self.get_mu_dim_map (idim)
      code_iter = 'i{}'.format (idim)
      if (ret != ''):
        ret += ', '
      if (mu_dim >= 0):
        np = self.PP.get_dim_size (mu_dim)
        ret += '(t{} * {} + i{})'.format (idim, np, idim)
      else:
        ret += code_iter
    return ret

  # Return the list of tile iterators as a comma-separated string list.
  # If the loop dimension is mapped, we return the processor coordinate
  # associated to it; Otherwise '0' is returned.
  def get_tile_iterator_str_list_complete (self, ref, PP):
    ret = ''
    used = 0
    for ii,dd in enumerate(self.dims):
      iter_name = self.dims[ii]
      if (ref.is_dim_used (iter_name)):
        if (used > 0):
          ret += ', '
        pdim = self.map[ii]
        if (pdim >= 0):
          # If the dimension is mapped, then the operator will only access
          # a tile within its slice. Return tile offset '0' w.r.t to
          # the slice.
          array_pdim = ref.get_proc_map_by_dim_name (iter_name)
          if (array_pdim >= 0):
            if (pdim != array_pdim):
              print ("[ERROR@get_tile_iterator_str_list]: Processor dimension mismatch between operator {}({}) and {}[{}]".format (self.name, iter_name, ref.get_name (), iter_name))
              sys.exit (42)
            ret += '0'
          else:
            # Operator is mapped, array dimension is replicated. 
            # Hence, we only access the slice corresponding to the processor
            # coordinate.
            ret += PP.get_processor_coordinate_variable (pdim)
        else:
          # The operator has a loop dimension unmapped. We will access the full
          # slice. Hence, return the tile iterator corresponding to the loop.
          ret += 't{}'.format (ii)
        used += 1
    return ret

  def assemble_list (self, tup):
    ret = ''
    for tt in tup:
      if (ret != ''):
        ret += ', '
      ret += tt
    return ret

  ## statement.permute_tile_access (): Must be used from the producer (generator)
  ## of an array.
  def permute_tile_access (self, ref, tiles, is_acum):
    ## Not used.
    if (is_acum):
      return tiles
    num_data_dim = ref.get_num_dim ()
    if (num_data_dim == 1):
      return tiles
    pi_map = ref.get_pi_map ()
    mu_map = self.map 
    num_proc_dim = self.PP.get_num_dim ()
    tup = tiles.split(', ')
    if (num_data_dim == 2):
      if (len (tup) != 2):
        print ("did not find 2 dimensions")
        sys.exit (42)
      if (len (pi_map) != 2):
        print ("pi not find 2 dimensions")
        sys.exit (42)
      if (len (mu_map) != 2):
        print ("mu not find 2 dimensions")
        sys.exit (42)
      print ("\t\t Input tile order: {} - Output tile order: {}, mu={}, pi={}, ntpd[0]={}".format (tiles, self.assemble_list ([tup[1],tup[0]]),  mu_map[1], pi_map[1], self.ntpd[0]))
      mu_val = mu_map[1]
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1):
        return self.assemble_list ([tup[1],tup[0]])
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1 and self.ntpd[0] > 1):
        return self.assemble_list ([tup[1],tup[0]])
      if (num_proc_dim == 1 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1):
        return self.assemble_list ([tup[1],tup[0]])
      mu_val = mu_map[1]
      if (num_proc_dim == 3 and pi_map[1] == -1 and mu_val >= 1 and self.PP.get_dim_size (mu_val) > 1 and self.ntpd[0] > 1):
        return self.assemble_list ([tup[1],tup[0]])
      return tiles
    if (num_data_dim == 3):
      print ("\t\t Input tile order (3DA): {} :: {} - Output tile order: {}, mu={}, pi={}".format (tiles, ref.get_name (), self.assemble_list ([tup[1],tup[0],tup[2]]),  mu_map[1], pi_map[1]))
      mu_val = mu_map[1]
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[1],tup[0],tup[2]])
      if (num_proc_dim == 1 and pi_map[1] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[1],tup[0],tup[2]])
      mu_val = mu_map[2]
      print ("\t\t Input tile order (3DB): {} :: {} - Output tile order: {}, mu={}, pi={}".format (tiles,  ref.get_name (), self.assemble_list ([tup[2],tup[0],tup[1]]),  mu_map[2], pi_map[2]))
      if (num_proc_dim == 2 and pi_map[2] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[2],tup[0],tup[1]])
      if (num_proc_dim == 1 and pi_map[2] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[2],tup[0],tup[1]])
      return tiles
    if (num_data_dim == 4):
      ## TODO: Pending to implement cases (1d, 2d and 3d arrays)
      return 'ERROR'
    return 'ERROR'
      
        
      

  ## Return the iteration space dimension used in position adim of the given ref.
  def get_dim_used_in_ref_dim (self, ref, adim):
    iter_name = ref.get_iter_name (adim)
    for ii in self.dims:
      if (iter_name == self.dims[ii]):
        return ii
    return -1

  # Return the permutation vector of an array reference.
  # This function makes sense when the @self instance is the 
  # producer of the array being accessed by @ref.
  def get_permutation_vector_from_map (self, ref, PP):
    ## Permutation vectors removed in lieu of including tile header and using tile maps.
    ret = [-2] * ref.get_num_dim ()
    left_i = set()
    left_p = set()
    for dd in self.dims:
      left_i.add (dd)
    for pp in range(PP.get_num_dim ()):
      left_p.add (pp)
    n_used = 0
    print ("PI-map of reference {}: {}".format (ref.get_name (), ref.get_pi_map ()))
    print ("MU-map of statement {}: {}".format (self.get_name (), self.get_mu_map ()))
    for dd in self.dims:
      iter_name = self.dims[dd]
      stmt_pdim = self.map[dd]
      if (stmt_pdim >= 0):
        print ('PV at {}, setting entry {} to {}'.format (self.name, dd, stmt_pdim))
        ret[dd] = stmt_pdim
        n_used += 1
        left_i.remove (dd)
        left_p.remove (stmt_pdim)
    print ("Pending processor dimensions: {}".format (left_p))
    print ("Unmapped iteration-space dimensions: {}".format (left_i))
    if (len(left_p) == 1):
      pending_i = next(iter(left_i))
      pending_p = next(iter(left_p))
      # The below condition returns an empty perm-vec when the 
      # missing processor dimension is not the first one and not a degenerate
      # processor dimension (of length 1).
      if (pending_i > 0 and PP.get_dim_size (pending_p) == 1): 
        return None
      if (pending_i >= 0 and pending_p >= 0):
        ret[pending_i] = pending_p
        left_i.remove (pending_i)
        left_p.remove (pending_p)
    # Shouldn't have pending dimensions.
    if (len(left_p) > 0):
      return None
    return ret

  def get_tile_iterator (self, idim):
    varname = 't{}'.format (idim)
    return varname

  def get_block_iterator (self, idim):
    varname = 'b{}'.format (idim)
    return varname

  def get_point_iterator (self, idim):
    varname = 'i{}'.format (idim)
    return varname

  def get_used_tile_iterator_list (self, ref):
    ret = ''
    adims = ref.get_dims ()
    for dd in adims:
      iter_name = adims[dd]
      for ii in self.dims:
        if (iter_name == self.dims[ii]):
          if (ret != ''):
            ret += ', '
          ret += self.get_tile_iterator (ii)
    return ret

  ## statement.
  def get_constant_tile_iterator_list (self, ref, constant):
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (ref.is_dim_used (iter_name)):
        if (ret != ''):
          ret += ', '
        ret += str(constant)
    return ret

  # @Statement: Return a comma-separated list of used tile iterators.
  # NOTE: Function get_tile_iterator_str_list_complete() performs a more 
  # complex job, which is to inline the iterator valued into the expression.
  def get_tile_iterator_str_list (self, ref, PP, producers, is_acum):
    ret = ''
    used = 0
    if (self.is_data_generator () or producers == None):
      for ii in range(ref.get_num_dim ()):
        idim = self.get_dim_used_in_ref_dim (ref, ii)
        if (used > 0):
          ret += ', '
        ret += self.get_tile_iterator (idim)
        used += 1
      return ret
    ttc = self.collect_tile_trip_counts (producers)
    comm_type = self.determine_communication_type (ref, PP)
    is_outgoing = False
    nacc = len(self.accs)
    if (not self.is_data_generator () and not self.is_data_sink ()
        and ref.get_name () == self.accs[nacc-1].get_name ()):
      is_outgoing = True
    is_allgather = is_outgoing and comm_type == COMM_TYPE_LOCAL_SLICE
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      if (idim >= 0):
        ## Next, determine if a tile iterator is 'local'. If that's the case,
        ## the data space dimension is effectively distributed, and the tile 
        ## index offset associated to the dimension becomes zero. We enforce
        ## this by shifting the tile bound by a block multiple
        mu_pdim = self.map[idim]
        iter_name = self.dims[idim]
        shift = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        num_proc = 1
        if (mu_pdim >= 0):
          trip = None
          if (iter_name in ttc):
            trip = ttc[iter_name]
          loop_lb = self.build_l2_loop (idim, trip, False, L2_LOOP_GENMODE_LB, producers)
          shift = ' - {}'.format (loop_lb)
          num_proc = int(self.PP.get_dim_size (mu_pdim))
        # If we only have one processor, shifting doesn't make sense.
        if (num_proc == 1):
          shift = ''
        if (ref.get_pi_dim_map (ii) < 0 and not is_allgather): # array is not partitioned along this dimension
          shift = ''
        if (used > 0):
          ret += ', '
        # If it's an accumulation loop and if the data-space dimension is unmapped, we proceed
        # to cancel shift as well, since we have to accumulate everything from the
        # temporary buffer.
        if (is_acum and ref.get_pi_dim_map (ii) < 0):
          shift = ''
        ret += self.get_tile_iterator (idim) + shift
        used += 1
    return ret

  # Return a reordered list of tile iterators.  This routine is normally
  # invoked when the lexico-positivity of an access is False. When the
  # iterators are not reordered we return None.
  def get_reordered_tile_iterator_str_list (self, ref, PP, producers, use_full_extent = False):
    if (not ref.get_name () in producers):
      print ('[ERROR@get_reordered_tile_iterator_str_list]: Found un-produced array {}'.format (ref.get_name ()))
      sys.exit (42)
    prod = producers[ref.get_name ()]
    permvec = prod.get_permutation_vector_from_map (ref, PP)
    if (permvec == None):
      print ('\t[INFO@get_reordered_tile_iterator_str_list]: returning empty perm-vec for ref {} @ stmt {}.'.format (ref.get_name (), self.name))
      return None
    temp = [None] * ref.get_num_dim ()
    ttc = self.collect_tile_trip_counts (producers)
    for dd in self.dims:
      iter_name = self.dims[dd]
      adim = ref.get_dim_if_used (iter_name)
      # Next, determine if a tile iterator is 'local'. If that's the case,
      # the data space dimension is effectively distributed, and the tile 
      # index offset associated to the dimension becomes zero. We enforce
      # this by multiplying by a zero factor.
      mu_pdim = self.map[dd]
      factor = ''
      match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
      is_subset_dim = ref.is_mu_map_dim_strict_subset_of_pi (iter_name, mu_pdim)
      prod_idims = prod.get_dims ()
      tile_size = 1
      if (adim >= 0):
        iter_name_in_prod = prod_idims[adim]
        if (iter_name_in_prod in ttc):
          tile_size = ttc[iter_name_in_prod]
      # Below: match_dim addresses the case of mu and pi maps matching in full.
      # When this occurs, computation becomes relative to this loop dimensions becomes
      # local, and we nullify these iterators by appending a '* 0' factor. The 
      # assumption here is that the corresponding buffer is just big enough
      # to fit the needed data.
      # The second case of the 'or' arises from a special case where implicit
      # tiling took place, likely from the generator, and the array still being
      # all-gathered along this dimension. This means we have the full slice along
      # such dimension, but it may have been reorganized with the all-gather.
      if (match_dim or (not use_full_extent and is_subset_dim and int(tile_size) > 1)):
        factor = ' * 0'
      if (adim >= 0):
        new_place = permvec[adim]
        temp[new_place] = self.get_tile_iterator (dd) + factor
    print ("=====> Reodered tile iterator list for reference {}[{}] @ statement {}({}): {}".format (ref.get_name (), ref.get_dims (), self.name, self.dims, temp))
    if (temp == None):
      return None
    ret = ''
    for tt in temp:
      if (ret != ''):
        ret += ', '
      if (tt == None):
        print ("[ERROR]: None entry found : {}".format (temp))
        return None
      ret += tt
    return ret

  def get_tile_trip_count_str_list (self, ref, PP, producers):
    nprocdim = PP.get_num_dim ()
    if (not ref.get_name () in producers):
      print ('[ERROR@get_tile_trip_count_str_list]: Found un-produced array {}'.format (ref.get_name ()))
      sys.exit (42)
    ttc = self.collect_tile_trip_counts (producers)
    ret = ''
    prod = producers[ref.get_name ()]
    permvec = prod.get_permutation_vector_from_map (ref, PP)
    temp = None
    # NOTE for below: the True branch of the below code will work
    # for the case when the dimensionality of the processor grid
    # matches that of the data space.
    if (permvec != None and ref.get_num_dim () == nprocdim):
      temp = [None] * ref.get_num_dim ()
      for idim in self.dims:
        iter_name = self.dims[idim]
        adim = ref.get_dim_if_used (iter_name)
        mu_pdim = self.map[idim]
        factor = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        degenerate_proc_dim = mu_pdim >= 0 and PP.get_dim_size (mu_pdim) == 1
        if (match_dim and not degenerate_proc_dim):
          factor = ' * 0'
        str_val = '1'
        if (iter_name in ttc):
          str_val = ttc[iter_name]
        if (adim >= 0):
          new_place = permvec[adim]
          if (new_place != None):
            temp[new_place] = str_val + factor
          else:
            temp[adim] = str_val + factor
    else:
      # The permutation vector is empty, so we just populate
      # the temporary vector with the collected trip counts (number of
      # tiles per dimension.
      temp = []
      for idim in self.dims:
        iter_name = self.dims[idim]
        adim = ref.get_dim_if_used (iter_name)
        if (adim < 0):
          continue
        mu_pdim = self.map[idim]
        factor = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        if (match_dim):
          factor = ' * 0'
        str_val = '1'
        if (iter_name in ttc):
          str_val = ttc[iter_name]
        str_val = str_val + factor
        temp.append(str_val)
    for ii,tt in enumerate(temp):
      if (ret != ''):
        ret += ', '
      if (tt == None):
        print ("[INFO]  Access funcf for reference {}: {}".format (self.name, self.dims))
        print ("[INFO]  Perm vec: {}".format (permvec))
        print ("[INFO]  Semi-reordered access: {}".format (temp))
        print ("[ERROR] None entry found : {}".format (temp))
        print ("[INFO]  entry {} : {}".format (ii,tt))
        return None
      ret += tt
    return ret

  # Return a list of tile sizes, possibly obtained from dividing
  # the loop trip count by the number of processors along a mapped dimension.
  def get_tile_size_str_list (self, ref, PP, producers):
    ttc = self.collect_tile_trip_counts (producers)
    ret = ''
    arr_dims = ref.get_dims ()
    for dd in range(len(arr_dims)):
      iter_name = arr_dims[dd]
      idim = self.get_dim_by_name (iter_name)
      mu_dim = self.map[idim]
      pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
      num_proc_dim = 1
      if (mu_dim >= 0):
        num_proc_dim = self.PP.get_dim_size (mu_dim)
      if (dd > 0):
        ret += ', '
      extent = ref.get_extent_as_str (dd)
      num_proc_along_dim = 1
      num_proc_cand = []
      if (iter_name in ttc):
        num_proc_cand.append (ttc[iter_name])
      num_proc_along_dim = PP.lcm () 
      tile_size = '{}(({}),{})'.format (DIMAGE_CEIL, extent, num_proc_along_dim)
      if (int(num_proc_along_dim) == 1):
        tile_size = '{}'.format (extent)
      ret += tile_size
    return ret

  ## @Statement:
  ## Return the list of tiles along each dimension.
  def get_tile_count_list (self, ref, PP, producers, is_allgat_out_slice):
    ttc = self.collect_tile_trip_counts (producers)
    ret = ''
    arr_dims = ref.get_dims ()
    for dd in range(len(arr_dims)):
      iter_name = arr_dims[dd]
      idim = self.get_dim_by_name (iter_name)
      mu_dim = self.map[idim]
      pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
      tc = self.get_number_tiles_along_dim (ref, idim, producers)
      if (pi_dim < 0 and not is_allgat_out_slice):
        # The above function is also used in build_l2_loop to determine the 
        # number of tiles that must be traversed by a loop. That traversal is
        # 'array independent (only depends on the mu mappings) and so in some
        # cases we really care about exclusively of the data mapping. This
        # condition fixes the case we really need the number of tiles along
        # a particular data-space dimension caused by a pi-mapping < 0 (unpartitioned).
        # We do this here since get_number_tiles_along_dim is used in other
        # places as well, while get_tile_count_list () is used exclusively for
        # generating the accesses to TILES.
        tc = self.PP.lcm ()
      ## Below we handle the case where a local computation is performed
      ## on a data slice that will then be all-gathered. In addition,
      ## the array is replicated along a particular dimension, and the 
      ## computation is mapped. Hence, the all-gather. In this case, 
      ## a temporary array has been allocated for the slice, and hence the
      ## the extents will be offset to the tiles locally computed.
      ## As a result, the number of effective tiles along the allocated slice
      ## has to be adjusted.
      if (mu_dim >= 0 and pi_dim < 0 and is_allgat_out_slice):
        denum = self.PP.get_dim_size (mu_dim)
        tc = '({}/{})'.format (PP.lcm (), denum)
      if (dd > 0):
        ret += ', '
      ret += '{} /* idim={} */'.format (tc, idim)
    return ret


  ## @Statement: Return the size (number of nodes) along a processor dimension.
  def get_num_proc_for_slice (self, ref, PP, producers, ttc):
    ## NOTE: ttc must have been populated.
    arr_dims = ref.get_dims ()
    ret = []
    prod = producers[ref.get_name ()]
    for dd in range(len(arr_dims)):
      iter_name = prod.get_dim_name (dd)
      mu_dim = prod.get_mu_dim_map (dd)
      pi_dim = ref.get_pi_dim_map (dd)
      num_proc_dim = 1
      print ("[INFO] =============> ttc={}, it={}, mu={}, pi={}".format (ttc, iter_name, mu_dim, pi_dim))
      if (mu_dim >= 0 and pi_dim < 0): #and iter_name in ttc):
       num_proc_dim = self.PP.get_dim_size (mu_dim)
      ret.append (num_proc_dim)
    if (len(ret) == 1):
      return '{} /* {} */'.format (ret[0], ret)
    print ("[INFO] =============> Collected processor dimensions: {}".format (ret))
    ret = sorted(ret)
    first = ret[0]
    all_equal = True
    non_degen = 0
    for ds in ret:
      if (ds > 1):
        first = ds
        non_degen += 1
    if (non_degen == 0):
      return '1'
    if (non_degen == 1):
      return first
    print ("Found two or more non-degenerate (>1) grid space-dimensions. Need max.common divisor")
    sys.exit (42)
    return '{} /* {} */'.format (ret[0], ret)

  # Statement.generate_access():
  # Generate a single (linearized) access reference.
  # Access type is determined by the number of tiles accessed in the 
  # current rank: TILE (single tile) and SLICE (multi-tile)
  # The SLICE case can account for multi-dimensional tiles.
  # Determining the access type considers the 4 (statement x array) scenarios:
  # case 1) mapped + mapped: single tile
  # case 2) unmapped + unmapped: single tile but which represents the full array
  # case 3) mapped + unmapped: all ranks have space for the full array, but 
  #   work only on their corresponding tile. Leads to reduce or allreduce 
  #   communication.
  # case 4) unmapped + mapped: all ranks attempt to work on the full spread 
  #   or slice of the array, hence require an all-gather first.
  def generate_access (self, ref, PP, producers, is_write = False, is_acum = False, use_full_extent = False, is_allgat_out_slice = False):
    acc_type = ref.get_access_type (self, PP, producers)
    ttc = self.collect_tile_trip_counts (producers)
    local_linear = (acc_type == ACC_TYPE_SLICE and is_write)
    macro = ''
    if (acc_type == ACC_TYPE_TILE):
      macro = DIMAGE_ACC_TILE
    elif (acc_type == ACC_TYPE_LIN or local_linear):
      macro = DIMAGE_ACC_LIN
    elif (acc_type == ACC_TYPE_SLICE):
      macro = DIMAGE_ACC_SLICE
    else:
      print ('[ERROR@generate_access]: Unexpected access type at statement {}, reference {}: {}'.format (self.name, ref.get_name (), acc_type))
      sys.exit (42)
    prod = producers[ref.get_name ()]
    ref_at_prod = prod.get_ref(0)
    is_lexpos = prod.is_mu_lexico_positive (ref, producers)
    special_lin = False 
    if (option_debug >= 10):
      print ("\n=====> SPECIAL LIN ({}) for ref={} at {}: rep={}, acc-type={}, lex_pos={}\n".format (special_lin, ref.get_name (), self.name, ref_at_prod.is_fully_replicated (), acc_type == ACC_TYPE_SLICE, is_lexpos))
    if (special_lin):
      macro = DIMAGE_ACC_LIN
    acc = '{}{}D'.format (macro, ref.get_ref_dim())
    acc += '('
    # Produce tile iterators: 0, t? or p?
    if (acc_type == ACC_TYPE_LIN or (local_linear) or special_lin):
      acc += ref.get_linearized_iterator_str_list (self, PP, producers, is_write, is_acum)
      acc += ', '
      acc += ref.get_mapped_array_extents_as_str_list (self, False, is_write)
    elif (acc_type == ACC_TYPE_TILE):
      # Will include only tile iterators used in the access function of the 
      # current reference.
      acc += self.get_constant_tile_iterator_list (ref, 0)
      acc += ', '
      acc += self.get_iterator_str_list_used_in_ref (ref)
      acc += ', '
      acc += self.get_tile_size_str_list (ref, PP, producers)
      acc += ', '
      ## NOTE: statement.get_tile_count_list () deals with several tile-extent cases,
      ## including all-gather-outgoing-slices.
      acc += self.get_constant_tile_iterator_list (ref, 1)
    elif (acc_type == ACC_TYPE_SLICE and not special_lin):
      reordered_tile_iter = self.get_reordered_tile_iterator_str_list (ref, PP, producers, use_full_extent)
      acc += self.get_tile_iterator_str_list (ref, PP, producers, is_acum)
      acc += ', '
      acc += self.get_sliced_iterator_str_list_used_in_ref (ref)
      acc += ', '
      acc += ref.get_mapped_array_extents_as_str_list (None, False, is_write)
      acc += ', '
      acc += '{}'.format (self.get_num_proc_for_slice (ref, PP, producers, ttc))
      prod = None
      if (ref.get_name () in producers):
        prod = producers[ref.get_name ()]
    else:
      print ("[ERROR] Unexpected access type")
      sys.exit (42)
    acc += ')'
    return acc

  ## Statement.generate_tile_fetch_code(): Insert RT call to find the data 
  ## tile corresponding to the passed tile iterators.
  ## 
  def generate_tile_fetch_code (self, df, ref, producers, is_acum, intermediate = None):
    tile_name = ref.get_tile_name (intermediate)
    tile_map = ref.get_tile_map_name (intermediate)
    tile_iters = self.get_used_tile_iterator_list (ref)  
    extent_list = ref.get_tile_extent_list ()
    source_array = ref.get_name ()
    is_out = self.is_outgoing_slice (ref)
    is_incom = self.is_incoming_slice (ref)
    if (option_debug >= 3):
      print ('[INFO] generate_tile_fetch_code (): stmt={}, ref={}, is_acum={}, interm={}, is_incom={}, is_outgoing={}'.format (self.name, ref.get_name (), is_acum, intermediate, is_incom, is_out))
    if (is_out and not is_acum and intermediate != None):
      tile_map = ref.get_tile_map_name () + '/* tile map same as target buffer */'
    if (is_out and not is_acum and self.is_true_communication (ref)):
      ref.set_use_slice (True)
      if (option_debug >= 5):
        print ("\t Slice name before : {}".format (source_array))
      source_array = ref.get_slice_varname (False)
      if (option_debug >= 5):
        print ("\t Slice name after: {}".format (source_array))
    if (is_incom and not is_acum):
      source_array = ref.get_slice_varname (True) + ' /* gathered slice */ '
    if (intermediate != None):
      source_array = intermediate 
    proc_size_list = self.PP.get_processor_geometry_list_from_map (ref.get_pi_map (), True)
    call = '{} *{} = {}_{}D ({}, {}, {}, {}, {});\n'.format (DIMAGE_DT, tile_name, DIMAGE_FETCH_TILE_FUNC, ref.get_num_dim (), source_array, tile_map, extent_list, tile_iters, proc_size_list)
    df.write (call)


  def generate_set_tile_coordinate (self, df, ref):
    source_array = ref.get_tile_name ()
    tile_iters = self.get_used_tile_iterator_list (ref)
    call = '{}_{}D({}, {});\n'.format (DIMAGE_SET_TILE_COORD_FUNC, ref.get_num_dim (), source_array, tile_iters)
    self.indent (df)
    df.write (call)

  ## @Statement.get_tile_pointer ():
  ## This function is called solely by statement.generate_dimage_operator_call ().
  def get_tile_pointer (self, ref, producers, is_allgat_slice, is_read = False):
    ret = ' &'
    refname = ref.get_tile_name ()
    if (is_allgat_slice and is_read):
      refname = ref.get_tile_name (ref.get_slice_varname (True))
    ret += refname
    ret += '['
    ret += '{}{}D ('.format (DIMAGE_TILE_POINTER, ref.get_num_dim ())

    prod = producers[ref.get_name ()]
    ref_at_prod = prod.get_ref(0)

    ret += self.get_constant_tile_iterator_list (ref, 0)
    ret += ', '
    ret += self.get_tile_size_str_list (ref, self.PP, producers)
    ret += ', '
    ret += self.get_constant_tile_iterator_list (ref, 1)
    ret += ')]'
    return ret



  def get_allred_intermediate (self, ref):
    return 'interm_{}_at_{}'.format (ref.get_name (), self.name )

  ## @Statement:
  def generate_write_to_file_arguments (self, ref, PP, is_operator = False):
    acc = ''
    if (is_operator):
      acc += DIMAGE_RANK_ARRAY 
    else:
      acc += DIMAGE_RANK_ARRAY
    acc += ', '
    acc += ref.get_name ()
    acc += ', '
    acc += ref.get_tile_extent_list ()
    acc += ', '
    acc += self.PP.get_processor_geometry_list_from_map (ref.get_pi_map (), False)
    return acc

  ## @Statement.generate_single_node_write_to_file_arguments ():
  ## Collect arguments for call to function that dumps an entire 
  ## -- single node -- matrix to a file.
  def generate_single_node_write_to_file_arguments (self, ref, PP):
    acc = ''
    acc += 'sna_' + ref.get_name ()
    acc += ', '
    acc += ref.get_dimension_size_as_str_list (self, PP, ALLOC_MODE_FULL)
    return acc

  ## Statement: Generate the arguments for reading one or more tiles into a buffer.
  def generate_read_from_file_arguments (self, ref, PP, check_mode = DIMAGE_CHECK_NO_CHECK):
    acc = ''
    array_name = ''
    comment = ''
    comm_type = self.determine_communication_type (ref, PP)
    if (check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      array_name = ref.get_name_for_check ()
      comment = ' /* DIMAGE_CHECK_READ_REF_ARRAY */ '
    elif (check_mode == DIMAGE_CHECK_CALL_CHECK):
      ## Array name for correctness check.
      array_name = ref.get_tile_name ()
      comment = ' /* DIMAGE_CHECK_CALL_CHECK - get_tile_name () */ '
      if (ref.get_use_slice ()):
        array_name = ref.get_tile_name (ref.get_slice_varname (True)) + '/* fetched slice */'
        comment = ' /* DIMAGE_CHECK_CALL_CHECK - get_slice_varname (T) */ '
    elif (comm_type == COMM_TYPE_LOCAL):
      array_name = ref.get_name ()
      comment = ' /* COMM_TYPE_LOCAL */ '
    elif (self.is_true_communication (ref)):
      array_name = ref.get_slice_varname (False)
      comment = ' /* ITC */ '
    else:
      array_name = ref.get_name ()
      comment = ' /* GRFFA - default */ '
    acc += array_name
    acc += ', '
    if (check_mode == DIMAGE_CHECK_CALL_CHECK):
      acc += ref.get_sna_ref_name ()
      acc += ', '
    acc += ref.get_array_extents_as_str_list ()
    acc += ', '
    acc += ref.get_dimension_size_as_str_list (self, PP, ALLOC_MODE_TILE)
    acc += ', '
    acc += self.get_tile_iterator_str_list (ref, PP, None, False)
    acc += comment
    return acc
    

  # Statement: Generate the right-hand expression of the initialization statement.
  # Take into account if an array dimension has been distributed
  # or if its fully local.
  def create_dimension_expression (self, ref, PP, idim, pc):
    expr = ''
    if (pc >= 0):
      proc_coord = PP.get_processor_coordinate_variable (pc)
      tile_size = ref.get_dimension_size_as_str (self, idim, PP)
      tile_var = 't{}'.format (idim)
      iter_var = 'i{}'.format (idim)
      expr = '({} * {} + {})'.format (tile_var,tile_size,iter_var)
    else:
      iter_var = 'i{}'.format (idim)
      expr = '({})'.format (iter_var)
    return expr
    
  # Statement: Generate a linearized global expression for a data generator.
  def generate_init_expression (self, PP):
    ref = self.accs[0]
    macro = DIMAGE_INIT_DIAG
    acc = '{}_{}D'.format (macro, ref.get_ref_dim())
    expr = '{}('.format (acc)
    for ii,pc in enumerate(self.map):
      dim_expr = self.create_dimension_expression (ref, PP, ii, pc)
      expr += dim_expr
      expr += ', '
    expr += ref.get_array_extents_as_str_list ()
    expr += ')'
    return expr

  ## Statement: 
  def compute_matmul (self, ref_out, ref_in, ref_ker):
    N0 = int (ref_out.get_extent_as_str (0))
    N1 = int (ref_out.get_extent_as_str (1))
    N2 = int (ref_ker.get_extent_as_str (0))
    mat_out = ref_out.get_data () 
    mat_in = ref_in.get_data () 
    mat_ker = ref_ker.get_data () 
    for ii in range(N0):
      for jj in range(N1):
        for kk in range (N2):
          mat_out[ii * N1 + jj] += mat_in[ii * N2 + kk] * mat_ker[kk * N1 + jj]
    self.write_matrix_to_file (mat_out, ref_out.get_name (), N0, N1)

  def get_operator_c_filename (self):
    operator_filename = '{}.dimage-op.c'.format (self.name)
    return operator_filename 

  def get_operator_bin_filename (self):
    operator_filename = self.get_operator_c_filename ()
    bin_filename = re.sub ("\.c",".exe", operator_filename)
    return bin_filename

  # Generate a baseline C-implementation.
  def compute_operator (self, op_refs, init_val = 1.0):
    operator_filename = self.get_operator_c_filename ()
    rcf = open(operator_filename, 'w')
    indent = '  '
    rcf.write ('#include "dimage-rt.h"\n')
    rcf.write ('int {}[] = {};\n'.format (DIMAGE_GRID_DIMS, '{1,1,1,1,}'))
    rcf.write ('int main () {\n')
    trips = {}
    for dd in self.dims:
      iter_name = self.dims[dd]
      for ref in op_refs:
        if (not iter_name in trips and ref.is_dim_used (iter_name)):
          ub = ref.get_array_extent_by_dim_name (iter_name)
          trips[iter_name] = ub
    if (not self.is_data_generator ()):
      for ref in op_refs:
        data_source = ref.get_matrix_filename ()
        array_size = ref.get_array_size_as_product_str ()
        rcf.write ('{}{} * {} = read_matrix_from_file (\"{}\", {});\n'.format(indent, DIMAGE_DT, ref.get_name (), data_source, array_size))
    for it in self.dims:
      iter_name = self.dims[it]
      rcf.write ('{}int {};\n'.format (indent, iter_name))
    depth = 1
    if (not self.is_data_sink () and not self.is_data_generator ()):
      for it in self.dims:
        iter_name = self.dims[it]
        ub = trips[iter_name]
        rcf.write ('{}for ({} = 0; {} < {}; {}++) {}\n'.format (indent * depth, iter_name, iter_name, ub, iter_name, '{'))
        depth += 1
      stmt_body = ''
      nref = len(op_refs)
      for ii,ref in enumerate(op_refs):
        if (ii == 0):
          continue
        access = ref.gen_canonical_access ()
        if (ii > 1):
          stmt_body += ' * '
        stmt_body += access
      write_access = op_refs[0].gen_canonical_access ()
      stmt_body = '{}{} += {};\n'.format (indent * depth, write_access, stmt_body)
      rcf.write (stmt_body)
      for it in self.dims:
        depth -= 1
        rcf.write ('{}{}\n'.format (indent * depth, '}'))
    out_size = op_refs[0].get_array_size_as_product_str ()
    mat_dim = op_refs[0].get_num_dim ()
    extents = op_refs[0].get_array_extents_as_str_list ()
    ## Generate code for logging, and free the arrays
    if (not self.is_data_sink () and not self.is_data_generator ()):
      data_sink = op_refs[0].get_matrix_filename (self.name)
      rcf.write ('{}{}_{}D (\"{}\", {}, {});\n'.format(indent, WRITE_MATRIX_TO_FILE, mat_dim, data_sink, op_refs[0].get_name (), extents))
      for ii,ref in enumerate(op_refs):
        rcf.write ('{}free ({});\n'.format (indent, ref.get_name ()))
    else:
      data_sink = op_refs[0].get_matrix_filename ()
      rcf.write ('{}generate_datafile_{}D (\"{}\", {}, {});\n'.format(indent, mat_dim, data_sink, extents, init_val))
    rcf.write ('{}return 0;\n'.format (indent))
    rcf.write ('}')
    rcf.close ()

  def write_matrix_to_file (self, mat, name, N0, N1):
    fmat = open ('{}-ref.mat'.format (name), 'w') 
    for ii in range(N0):
      for jj in range(N1):
        fmat.write ('{:.6f} '.format (mat[ii * N1 + jj]))
      fmat.write ('\n')
    fmat.close ()  

  ## statement.is_outgoing_slice ():
  def is_outgoing_slice (self, ref):
    is_write = self.is_write_ref (ref)
    if (not is_write):
      return False
    out_comm_type = self.determine_communication_type (ref, self.PP)
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (out_comm_type == COMM_TYPE_GATHER_SLICE, out_comm_type == COMM_TYPE_ALLRED, out_comm_type == COMM_TYPE_LOCAL_SLICE, self.is_true_communication (ref))
    return outgoing_slice_comm_type 

  ## Statement: Determine whether a reference at the current operator requires 
  ## incoming communication.
  def is_incoming_slice (self, ref):
    is_write = self.is_write_ref (ref)
    if (is_write):
      return False
    in_comm_type = self.determine_communication_type (ref, PP)
    is_true_comm = self.is_true_communication (ref)
    return in_comm_type == COMM_TYPE_GATHER_SLICE and is_true_comm
    ## The Below code is never reachable.
    out_comm_type = self.determine_communication_type (ref, self.PP)
    is_outgoing_gather = out_comm_type == COMM_TYPE_GATHER_SLICE
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (ref)
    is_outgoing_allreduce = out_comm_type == COMM_TYPE_ALLRED
    is_outgoing_local_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (is_outgoing_gather, is_outgoing_allreduce, is_outgoing_local_slice, is_true_comm)

  ## Statement: Generate the statement body for the three types of statements 
  ## i.e., regular, generator or sink.
  def generate_statement (self, df, PP, producers, mrap, indent, check_mode = DIMAGE_CHECK_NO_CHECK):
    if (self.is_data_generator ()):
      array = self.accs[0]
      if (DO_REF and False):
        array.gen_matrix_data ()
      mrap[array.get_name ()] = array
      rff_args = self.generate_read_from_file_arguments (array, PP)
      rff_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      buffer_name = array.get_matrix_filename ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (rff_func, num_array_dim, buffer_name, rff_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      ## Generate the call to read_from_file_tile to load the reference tile.
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      buffer_name = array.get_sna_reference_filename ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, buffer_name, acf_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_CALL_CHECK):
      ## Call to function for correctness check.
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = ARRAY_CHECK_FUNC
      num_array_dim = array.get_num_dim ()
      check_filename = '{}_at_{}'.format (array.get_name (), self.get_name ())
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, check_filename, acf_args, DIMAGE_REFBLOCK_COUNT)
    if (self.is_data_sink ()):
      array = self.accs[0]
      sink_varname = array.get_name ()
      if (array.get_use_slice ()):
        sink_varname = array.get_slice_varname (True)
      wtf_args = self.generate_write_to_file_arguments (array, PP)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      return '//{}_tile{}D(\"data_{}\", {});\n'.format (wtf_func, num_array_dim, sink_varname, wtf_args)
    num_accs = len(self.accs)
    if (num_accs != 3):
      print ('[ERROR@generate_statement]: Expected only 3 array references. Found {} instead.'.format (num_accs))
      sys.exit (42)
    in_ref = self.accs[0]
    ker_ref = self.accs[1]
    out_ref = self.accs[2]
    if (not in_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (in_ref.get_name ()))
      sys.exit (42)
    if (not ker_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (ker_ref.get_name ()))
      sys.exit (42)
    if (not out_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (out_ref.get_name ()))
      sys.exit (42)
    lp_in_ref = producers[in_ref.get_name ()].is_mu_lexico_positive (in_ref, producers)
    lp_ker_ref = producers[ker_ref.get_name ()].is_mu_lexico_positive (ker_ref, producers)
    lp_out_ref = producers[out_ref.get_name ()].is_mu_lexico_positive (out_ref, producers)
    # Compute: fetch most recently produced arrays.
    mrap_in_ref = None
    mrap_ker_ref = None
    mrap_out_ref = None
    if (in_ref.get_name () in mrap):
      mrap_in_ref = mrap[in_ref.get_name ()]
    if (ker_ref.get_name () in mrap):
      mrap_ker_ref = mrap[ker_ref.get_name ()]
    if (out_ref.get_name () in mrap):
      mrap_out_ref = mrap[out_ref.get_name ()]
    if (DO_REF and False):
      mrap_in_ref.show_data ()
      mrap_ker_ref.show_data ()
      mrap_out_ref.show_data ()
    if (DO_REF and False):
      self.compute_matmul (mrap_out_ref, mrap_in_ref, mrap_ker_ref)
      print ("Array {} after matmul...".format (mrap_out_ref.get_name ()))
    if (DO_REF and False):
      mrap_out_ref.show_data ()
    if (option_debug >= 4):
      print ("[INFO] Replacing old array {} in mrap ...".format (mrap_out_ref.get_name ()))
    mrap[mrap_out_ref.get_name ()] = mrap_out_ref

    in_comm_type = self.determine_communication_type (in_ref, PP)
    ker_comm_type = self.determine_communication_type (ker_ref, PP)
    out_comm_type = self.determine_communication_type (out_ref, PP)
    
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (out_ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (out_comm_type == COMM_TYPE_GATHER_SLICE, out_comm_type == COMM_TYPE_ALLRED, out_comm_type == COMM_TYPE_LOCAL_SLICE, self.is_true_communication (out_ref))

    in_acc = self.generate_access (in_ref, PP, producers)
    ker_acc = self.generate_access (ker_ref, PP, producers)
    out_acc = self.generate_access (out_ref, PP, producers, True, False, False, is_allgat_out_slice)
    ## We store this here to avoid recomputing a bunch of information
    ## later needed in generating the pointer-access for external kernels.
    out_ref.set_is_allgat_out_slice (is_allgat_out_slice)
    out_ref.set_precollective_buffer_access (out_acc)

    in_ref_was_allgat =  in_ref.get_is_allgat_in_slice ()  
    ker_ref_was_allgat = ker_ref.get_is_allgat_in_slice () 
    out_ref_was_allgat = producers[out_ref.get_name ()].was_allgathered (out_ref, self)

    comment_in  = '' # /* Lexico+ : {} - ag = {} */'.format (lp_in_ref, in_ref_was_allgat)
    comment_ker = '' # /* Lexico+ : {} - ag = {} - */'.format (lp_ker_ref, ker_ref_was_allgat)
    comment_out = '' #/* Lexico+ : {} - ag = {} - */'.format (lp_out_ref, out_ref_was_allgat)

    ret = ''
    ret += '{}[{}] += {} \n'.format (out_ref.get_tile_name (), out_acc, comment_out)
    ret += indent + '  '
    left_buf = in_ref.get_tile_name ()
    if (in_ref_was_allgat):
      left_buf = in_ref.get_tile_name (in_ref.get_slice_varname (True))
    ret += '{}[{}] * {} \n'.format (left_buf, in_acc, comment_in)
    ret += indent + '  '
    right_buf = ker_ref.get_tile_name ()
    if (ker_ref_was_allgat):
      right_buf = ker_ref.get_tile_name (ker_ref.get_slice_varname (True))
    ret += '{}[{}]; {} \n'.format (right_buf, ker_acc, comment_ker)
    return ret


  def generate_statement_generic (self, df, PP, producers, mrap, indent, check_mode = DIMAGE_CHECK_NO_CHECK):
    if (self.is_data_generator ()):   
      array = self.accs[0]
      if (DO_REF and False):
        array.gen_matrix_data ()
      mrap[array.get_name ()] = array
      rff_args = self.generate_read_from_file_arguments (array, PP)
      rff_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (rff_func, num_array_dim, array.get_matrix_filename (), rff_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      buffer_name = array.get_sna_reference_filename ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, buffer_name, acf_args, DIMAGE_BLOCK_COUNT )
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_CALL_CHECK):
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = ARRAY_CHECK_FUNC
      num_array_dim = array.get_num_dim ()
      check_filename = '{}_at_{}'.format (array.get_name (), self.get_name ())
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, check_filename, acf_args, DIMAGE_REFBLOCK_COUNT)
    if (self.is_data_sink ()):   
      array = self.accs[0]
      wtf_args = self.generate_write_to_file_arguments (array, PP)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      return '//{}_tile{}D(\"data_{}\", {});\n'.format (wtf_func, num_array_dim, array.get_name (), wtf_args)
    operator_ref = []
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    if (not out_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (out_ref.get_name ()))
      sys.exit (42)
    lp_out_ref = producers[out_ref.get_name ()].is_mu_lexico_positive (out_ref, producers)
    mrap_out_ref = None
    if (out_ref.get_name () in mrap):
      mrap_out_ref = mrap[out_ref.get_name ()]
    operator_ref.append (out_ref)
    ret = ''
    out_comm_type = self.determine_communication_type (out_ref, PP)
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (out_ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    out_acc = self.generate_access (out_ref, PP, producers, True, False, False, is_allgat_out_slice)
    out_ref.set_is_allgat_out_slice (is_allgat_out_slice)
    out_ref.set_precollective_buffer_access (out_acc)
    was_allgat =  producers[out_ref.get_name ()].was_allgathered (out_ref, self)
    comment_out = '' # /* Lexico+ : {} - ag = {} - */'.format (lp_out_ref, was_allgat)
    ret += '{}[{}] += {} \n'.format (out_ref.get_tile_name (), out_acc, comment_out)
    print ("Array {} before operator ...".format (mrap_out_ref.get_name ()))
    mrap_out_ref.show_data ()
    for refid in range(num_accs-1):
      in_ref = self.accs[refid]
      if (not in_ref.get_name () in producers):
        print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (in_ref.get_name ()))
        sys.exit (42)
      lp_in_ref = producers[in_ref.get_name ()].is_mu_lexico_positive (in_ref, producers)
      was_allgat = in_ref.get_is_allgat_in_slice () 
      # Compute: fetch most recently produced arrays.
      mrap_in_ref = None
      if (in_ref.get_name () in mrap):
        mrap_in_ref = mrap[in_ref.get_name ()]
      operator_ref.append (in_ref)
      mrap_in_ref.show_data ()
      in_comm_type = self.determine_communication_type (in_ref, PP)
      in_acc = self.generate_access (in_ref, PP, producers)
      comment_in = '' #'/* Lexico+ : {} - ag = {} - */'.format (lp_in_ref, was_allgat)
      ret += indent + '  '
      mid_op = '*'
      if (refid == num_accs - 2):
        mid_op = ';'
      in_buf = in_ref.get_tile_name ()
      if (was_allgat):
        in_buf = in_ref.get_tile_name (in_ref.get_slice_varname (True))
      ret += '{}[{}] {} {} \n'.format (in_buf, in_acc, mid_op, comment_in)
    if (DO_REF):
      self.compute_operator (operator_ref)
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.dump_generated_tile (df, PP)
      ref.return_allocated (df)
    mrap_out_ref.show_data ()
    if (option_debug >= 5):
      print ("Replacing old array {} in mrap ...".format (mrap_out_ref.get_name ()))
    mrap[mrap_out_ref.get_name ()] = mrap_out_ref
    return ret

  ## @Statement: Generate the call to an external DIMAGE operator
  def generate_dimage_operator_call (self, df, PP, producers, mrap, indent):
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    is_allgat_out_slice = out_ref.get_is_allgat_out_slice ()
    loop_depth = len(self.dims)
    call ='dimage_{} ('.format (self.name)
    if (self.kernel_name != None):
      call ='dimage_{} ('.format (self.kernel_name)
    call += self.get_tile_pointer (out_ref, producers, is_allgat_out_slice, False)
    for refid in range(num_accs-1):
      ref = self.accs[refid]
      is_in_slice = ref.get_is_allgat_in_slice ()
      call += ', \n'
      call += '{} '.format (indent)
      ## Always pass False as the third argument (is_allgat_out_slice), since
      ## it will be a read-array.
      call += self.get_tile_pointer (ref, producers, is_in_slice, True) 
    for idim in range(len(self.dims)):
      call += ', '
      ub = self.build_loop_structure (idim, PP, True, producers, False, False, True)
      call += '{} + 1'.format (ub)
    call += ', 1.0'
    call += ', 1.0'
    call += ');'
    return call
    

  def gencode_matrix_data_generator (self, init_val):
    operator_ref = []
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    operator_ref.append (out_ref)
    for refid in range(num_accs-1):
      in_ref = self.accs[refid]
      operator_ref.append (in_ref)
    self.compute_operator (operator_ref, init_val)

  # @Statement: 
  def generate_simple_increment_statement (self, df, ref, PP, producers, mrap, indent):
    out_acc = self.generate_access (ref, PP, producers, False, True, True)
    buffer_acc = ref.get_precollective_buffer_access ()
    interm = 'tile_' + self.get_allred_intermediate (ref)
    ret = ''
    ret += '{}[{}] += \n{}{}{}[{}];'.format (ref.get_tile_name (), out_acc, indent, indent, interm, out_acc) #buffer_acc)
    return ret

  # @Statement: 
  def insert_omp_pragmas (self, df, indent):
    if (self.is_data_generator () or self.is_data_sink ()):
      return
    df.write (indent)
    iter_list = self.get_iterator_str_list ()
    df.write ('#pragma omp parallel for private({})\n'.format (iter_list))

  # @Statement: 
  # Construct the loop body, tile and point loops, as well as the statement
  # body for the current operator. For generators and data sinks, will use
  # the dimage-rt API to load and write data tiles in the correct order.
  # For computational operators, build_operator_body will decide different
  # access types for each reference. This is done by querying the @producers
  # dictionary to find out the statement that produced, and hence defined,
  # the layout of a data array.
  def build_operator_body (self, df, PP, producers, mrap):
    indent = BASE_INDENT
    depth = 1
    self.start_computation_timer (df, indent)
    self.ntpd = [] # number of tiles per dimension
    # Generate tile loops
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, False, producers, False, False)
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    ttc = self.collect_tile_trip_counts (producers)
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      if (dim_name in ttc):
        trip = ttc[dim_name]
      loop = self.build_l2_loop (level, trip, False, L2_LOOP_GENMODE_FULL, producers)
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      level += 1
      depth += 1
    for tt in ttc:
      line = '{}// dim {} : {}\n'.format (indent, tt, ttc[tt])
      df.write (line)
    ## Insert data tile fetching code
    if (not self.is_data_generator ()):
      for ref in self.accs:
        df.write (indent)
        ## Need to determine if/when intermediates are used. These are the 
        ## result of incoming all-gathers.
        intermediate = None
        if (ref.get_use_slice ()):
          intermediate = ref.get_slice_varname (True)
        self.generate_tile_fetch_code (df, ref, producers, False, intermediate)
        if (self.is_output_array (ref)):
          df.write (indent)
          self.generate_set_tile_coordinate (df, ref)
    if (not self.is_data_sink () and not self.is_data_generator ()):
      df.write (indent)
      df.write ('#ifdef DIMAGE_KERNEL_LOOP\n')
    self.insert_omp_pragmas (df, indent)
    # Generate point loops
    if (not self.is_data_sink () and not self.is_data_generator ()):
      for idim in self.dims:
        loop = self.build_loop_structure (idim, PP, True, producers, False, False)
        line = '{} {}\n'.format (loop, '{')
        df.write (indent)
        df.write (line)
        indent += BASE_INDENT
        depth += 1
    stmt_body = ''
    if (len(self.accs) == 3):
      stmt_body = self.generate_statement (df, PP, producers, mrap, indent)
    else:
      stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent)
    ## Reset the block count for data sinks: only one block is used when
    ## performing a check between the distributed tensor and the reference one.
    if (self.is_data_sink ()):
      ## Data sink check.
      ref = self.accs[0]
      line = '{}{} *{} = ({} + {});\n'.format (indent, DIMAGE_DT, ref.get_sna_ref_name (), ref.get_name_for_check (), DIMAGE_TILE_HEADER_MACRO)
      df.write (line)
      df.write ('{}{} = 0;\n'.format (indent, DIMAGE_BLOCK_COUNT))
      if (self.is_data_sink ()):
        df.write ('{}{} = 0;\n'.format (indent, DIMAGE_REFBLOCK_COUNT))
    df.write (indent)
    df.write (stmt_body)
    if (DIMAGE_OPTION_DO_CHECK and self.is_data_sink ()):
      if (len(self.accs) == 1):
        df.write (indent)
        df.write ('/* DIMAGE_OPTION_DO_CHECK */\n')
        stmt_body = self.generate_statement (df, PP, producers, mrap, indent, DIMAGE_CHECK_READ_REF_ARRAY) 
        df.write (indent)
        df.write (stmt_body)
        stmt_body = self.generate_statement (df, PP, producers, mrap, indent, DIMAGE_CHECK_CALL_CHECK)
        df.write (indent)
        df.write (stmt_body)
      else:
        stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent, DIMAGE_CHECK_READ_REF_ARRAY)
        df.write (indent)
        df.write (stmt_body)
        stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent, DIMAGE_CHECK_CALL_CHECK)
        df.write (indent)
        df.write (stmt_body)
    for lev in range(depth-1):
      if (not self.is_data_sink () and not self.is_data_generator () and lev == len(self.dims)):
        ## Make call to external DIMAGE operator         
        df.write (indent)
        df.write ('#else\n')
        df.write (indent)
        df.write ('// External DIMAGE operator\n')
        dimage_call = self.generate_dimage_operator_call (df, PP, producers, mrap, indent)
        df.write ('{}{}{}\n'.format (indent, indent, dimage_call))
        df.write (indent)
        df.write ('#endif\n')
      indent = BASE_INDENT * (depth-1)
      df.write (indent)
      df.write ('}\n')
      depth = depth - 1
    self.stop_computation_timer (df, indent)

  def declare_local_iterators (self, df):
    for dd in self.dims:
      line = 'int b{};\n'.format (dd)
      self.indent (df)
      df.write (line)
      line = 'int t{};\n'.format (dd)
      self.indent (df)
      df.write (line)
      line = 'int i{};\n'.format (dd)
      self.indent (df)
      df.write (line)

  ## @Statement:
  ## Do map(mu) <- map(pi) but only for generators that would normally require
  ## an all-gather.
  def statement_equalize_mu_to_pi_map (self, ref):
    if (self.statement_can_allgather_incoming (ref)):
      return
    for idim in self.dims:     
      dim_name = self.dims[idim]
      pival = ref.get_pi_by_dim_name_if_used (dim_name)
      muval = self.map[idim] 
      if (option_debug >= 4):
        print ('[INFO:mu<-pi] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
      # Normally, we would need to check that pival is not DIM_NOT_USED.
      # However, since we are in a generator, the number of iteration and
      # data space dimensions should be the same. In other words, all 
      # iteration space dimensions should always be used.
      self.map[idim] = pival

  def statement_can_allgather_incoming (self, ref):
    scenario = self.statement_is_unsupported_incoming_allgather (ref)
    return scenario == 0

  # @Statement:
  def statement_is_unsupported_incoming_allgather (self, ref):
    # Statement must be a generator
    if (self.is_compute_statement ()):
      return 0
    # Traverse the iteration and data dimensions to find an iteration 
    # space dimension i and data space dimension d that are used in the 
    # reference and s.t. mu_i is unmapped and pi_d is mapped.
    # When mu_i is unmapped (< 0) and pi_d is mapped, we would normally need
    # an all-gather to make local all data needed by the generator.
    # However, generators cannot do all-gathers since, by definition, they
    # only produce data.
    for idim in self.dims:     
      dim_name = self.dims[idim]
      pival = ref.get_pi_by_dim_name_if_used (dim_name)
      muval = self.map[idim]
      if (pival >= 0 and muval < 0):
        return 1
      if (pival >= 0 and muval >= 0 and pival != muval):
        return 2
    return 0

  # @Statement:
  # Repair mapping for the following scenarios:
  # 1) dim(grid) > dim(tensor), e.g., a 3D grid on a 2D tensor. There has to be replication.
  # 2) mu unmapped (-1) and pi mapped on generators. A regular operator would have the ability
  #    to `fetch' the data from wherever it is, but generator can't do this since they are the
  #    ones setting the initial placement of the data.
  def statement_align_mu_mappings (self, PP):
    dim_grid = PP.get_num_dim ()
    dim_comp = len(self.dims) 
    if (dim_grid <= dim_comp and self.is_compute_statement ()):
      if (option_debug >= 3):
        print ('@ statement_align_mu_mappings(): Nothing to do for computation {}'.format (self.name))
      return
    grid_case = dim_grid > dim_comp
    generator_case = not self.is_compute_statement ()
    if (option_debug >= 3):
      if (grid_case):
        print ('[INFO:R1] dim(grid) = {} > {} = dim({})'.format (dim_grid, dim_comp, self.name))
      if (generator_case):
        print ('[INFO:R2] Generator / Sink case on {}', self.name)
    if (grid_case):
      for ii,aa in enumerate(self.refs):
        ref = self.refs[aa]
        for idim in self.dims:     
          dim_name = self.dims[idim]
          pival = ref.get_pi_by_dim_name_if_used (dim_name)
          muval = self.map[idim]
          if (muval >= 0 and pival >= 0 and muval != pival):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
          elif (muval < 0 and pival >= 0):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
          elif (muval >= 0 and pival < 0):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
    if (generator_case):
      for ii,aa in enumerate(self.refs):
        ref = self.refs[aa]
        if (not self.statement_can_allgather_incoming (ref)):
          scenario = self.statement_is_unsupported_incoming_allgather (ref)
          if (option_debug >= 4):
            print ('[INFO]: Aligning mappings of generators: mu^{} with pi^{} - Scenario {}'.format (self.name, ref.get_name (), scenario))
          self.statement_equalize_mu_to_pi_map (ref)


  # @Statement: 
  def report_mappings (self, AA, PP):
    line='{}:'.format (self.name)
    for idim in self.dims:
      mu = self.map[idim]
      if (mu < 0):
        mu = '*'
      if (idim > 0):
        line += ','
      line += str(mu)
    print (line)
    nref = len(self.refs)
    for ii,aa in enumerate(self.refs):
      ref = self.refs[aa]
      line = '  {}:'.format (ref.get_name ())
      pilist = ''
      for idim in self.dims:     
        dim_name = self.dims[idim]
        pival = ref.get_pi_by_dim_name_if_used (dim_name)
        if (pival == DIM_NOT_USED):
          continue
        if (pilist != ''):
          pilist += ','
        if (pival >= 0):
          pilist += str(pival)
        else:
          pilist += '*'
      print (line+pilist)
      dim_comp = len(self.dims)
      dim_grid = PP.get_num_dim ()
      for idim in self.dims:     
        dim_name = self.dims[idim]
        pival = ref.get_pi_by_dim_name_if_used (dim_name)
        if (pival == DIM_NOT_USED):
          continue
        muval = self.map[idim]
        tag = 'ERR'
        if (muval == pival and pival == -1):
          tag = '(case 1) Data replicated, work replicated '
        elif (muval == pival and pival != -1):
          tag = '(case 2) Data distributed, work distributed (and matched)'
        elif (muval >= 0 and pival < 0):
          tag = '(case 3) Data replicated, work distributed'
        elif (muval < 0 and pival >= 0):
          collective_type = ''
          if (self.is_data_sink ()):
            collective_type = 'allgather [read]'
          elif (self.is_data_generator ()):
            collective_type = 'allbroadcast [write]'
          elif (ii == nref - 1):
            collective_type = 'allbroadcast [write]'
          else:
            collective_type = 'allgather [read]'
          tag = '(case 4) Data distributed, work replicated (need {})'.format (collective_type)
        elif (muval >= 0 and pival >= 0 and muval != pival and dim_grid > dim_comp):
          tag = '(case 5) Grid dimensionality ({}) exceeds dimensionality of computation ({})'.format (dim_grid, dim_comp)
        else:
          tag = 'Unexpected case (mu={},pi={})'.format (muval,pival)
        print ('    {}: {}'.format (dim_name, tag))
        
  def describe_communication_vector (self, vec):
    line = ''
    for ct in vec:
      if (line != ''):
        line += ', '
      if (ct == COMM_TYPE_ALLRED):
        line += 'ALL_REDUCE'
      if (ct == COMM_TYPE_LOCAL):
        line += 'LOCAL_COMM'
      if (ct == COMM_TYPE_LOCAL_SLICE):
        line += 'LOCAL_SLICE'
      if (ct == COMM_TYPE_GATHER_SLICE):
        line += 'GATHER_SLICE'
    line = '\t[INFO] Communication vector: [' + line + ']'
    print (line)

  ## Statement.determine_communication_type():
  def determine_communication_type (self, ref, PP):
    ctypes = []
    grid_dim = PP.get_num_dim ()
    for idim in self.dims:
      dim_name = self.dims[idim]
      red_dim = self.get_mapped_reduction_dimension (ref, PP)

      pp_red_dim_size = 1  ## Size of 1 means really no reduction.
      if (red_dim >= 0):
        pp_red_dim = self.map[red_dim]
        pp_red_dim_size = PP.get_dim_size (pp_red_dim)
      if (red_dim >= 0 and red_dim == idim and pp_red_dim_size > 1):
        print ("{}({}={}) is reduction dimension".format (self.name, idim, dim_name))
        ctypes.append (COMM_TYPE_ALLRED)
        continue
      if (ref.is_dim_used (dim_name)):
        stmt_pdim = self.map[idim]
        ref_pdim = ref.get_proc_map_by_dim_name (dim_name)
        if (option_debug >= 3):
          print ("Scenario: {}({}={}->{}), {}[{}->{}]".format (self.name, idim, dim_name, stmt_pdim, ref.get_name (), dim_name, ref_pdim))
        if (stmt_pdim == ref_pdim): # perfect match, local comm.
          if (option_debug >= 3):
            print ("\t --> LOCAL COMM")
          ctypes.append (COMM_TYPE_LOCAL)
          continue
        if (ref_pdim == -1 and stmt_pdim >= 0): # ref is replicated, so we have what we need (access only a piece of it, though), all-reduce!
          ## Make the distinction here because data generators can only do all-gathers.
          if (self.is_data_generator ()):
            if (option_debug >= 3):
              print ("\t [INFO] Generator {} mapped (mu >= 0), data replicated (pi < 0). Will all-gather".format (self.name))
            ctypes.append (COMM_TYPE_LOCAL_SLICE)
            continue
          else: 
            if (option_debug >= 2):
              print ("\t ALL-REDUCE (mu-mapped, pi-replicated) each procs computes different. All-reduce needed (psize={}). (COMM_TYPE_ALLRED)".format (pp_red_dim_size))
            ctypes.append (COMM_TYPE_ALLRED)
            continue
        if (ref_pdim >= 0 and stmt_pdim == -1): # ref is distributed, but all statements access everything. Need AllGather.
          if (option_debug >= 3):
            print ("\t REPLICATED WORK, PARTITIONED DATA (COMM_TYPE_GATHER_SLICE)")
            print ("\t[WARNING@determine_communication_type:1]: Accessing non-local data with mappings (stmt={},ref={},dim={}). Will require AllGather.".format (self.name, ref.get_name (), dim_name))
          ctypes.append (COMM_TYPE_GATHER_SLICE)
          continue
        if (stmt_pdim != ref_pdim and ref_pdim >= 0 and stmt_pdim >= 0): 
          if (option_debug >= 3):
            print ("\tDIMENSIONS MAPPED AND CROSSED")
          print ("\t[ERROR@determine_communication_type:2]: Accessing non-local data with mappings (stmt={},ref={},dim={}). Dimensions crossed.".format (self.name, ref.get_name (), dim_name))
          ctypes.append (COMM_TYPE_GATHER_SLICE)
          continue
        if (ref_pdim == -1 and stmt_pdim == -1):
          if (option_debug >= 3):
            print ("\t ALL-REPLICATED / No comm")
          ctypes.append (COMM_TYPE_LOCAL)
          continue
        print ("\t[ERROR@determine_communication_type:3]: Unexpected combination of mappings")
        sys.exit (42)
        continue
    all_local = True
    n_local_slices = 0
    n_gather_slices = 0
    n_allred = 0
    n_p2p = 0
    for ct in ctypes:
      all_local = all_local and ct == COMM_TYPE_LOCAL
      if (ct == COMM_TYPE_ALLRED):
        n_allred += 1
      if (ct == COMM_TYPE_LOCAL_SLICE):
        n_local_slices += 1
      if (ct == COMM_TYPE_GATHER_SLICE):
        n_gather_slices += 1
      if (ct == COMM_TYPE_P2P):
        n_p2p += 1
    if (all_local):
      return COMM_TYPE_LOCAL
    if (n_allred >= 1): ## We can expect two or more reduction dimensions (e.g. mttkrp)
      return COMM_TYPE_ALLRED
    if (option_debug >= 2):
      self.describe_communication_vector (ctypes)
    if (n_p2p > 0):
      print ("\t[WARNING][determine_communication_type:5] Found p2p comm.type. Unsupported. Aborting ...")
      return COMM_TYPE_LOCAL
    if (n_local_slices > 0 and n_gather_slices > 0):
      print ("\t[ERROR][determine_communication_type:6]: Cannot require local ({}) and gather slices ({}) along different dimensions. Aborting ...".format (n_local_slices,n_gather_slices))
      sys.exit (42)
    if (n_local_slices > 0 and n_gather_slices == 0):
      return COMM_TYPE_LOCAL_SLICE
    if (n_local_slices == 0 and n_gather_slices > 0):
      return COMM_TYPE_GATHER_SLICE
    # Shouldn't get to this point.
    print ("\t[ERROR][determine_communication_type:7] Found p2p comm.type. Unsupported. Aborting ...")
    sys.exit (42)
    return COMM_TYPE_LOCAL_SLICE

  ## @Statement: print the mapping properties of a statement and a reference
  def generate_accessor_summary (self, df, ref):
    self.indent (df)
    df.write ('// Array {}[]: '.format (ref.get_name ()))
    df.write ('\n')
    self.indent (df)
    df.write ('// mu-vector: ')
    self.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// pi-vector: ')
    ref.pretty_print_map (df)
    df.write ('\n')

  ## Statement: 
  def statement_generate_incoming_communication (self, df, PP):
    if (self.is_data_generator ()):   
      ref = self.accs[0]
      self.indent (df)
      df.write ('// Generators don\'t require incoming communication\n')
      self.indent (df)
      df.write ('// Array {}[]: '.format (ref.get_name ()))
      df.write ('\n')
      self.indent (df)
      df.write ('// mu-vector: ')
      self.pretty_print_map (df)
      df.write ('\n')
      self.indent (df)
      df.write ('// pi-vector: ')
      ref.pretty_print_map (df)
      df.write ('\n')
      return
    acc_pos = 0
    slices = []
    for ref in self.accs:
      self.indent (df)
      df.write ("// **************************************************** \n")
      self.generate_accessor_summary (df, ref)
      comm_type = self.determine_communication_type (ref, PP)
      ref_slice = ref.reference_generate_incoming_communication (df, self, comm_type, PP)
      if (ref_slice != None):
        slices.append (ref_slice)
      acc_pos += 1
    return slices

  ## Statement.is_true_communication(): Determine whether the current statement
  ## will require communication under the give array mapping.
  ## True communication happens along a given dim in two cases:
  ## 1) k is used mu_k is mapped and pi_k is not mapped: that means that computation
  ##    is local, but that data is replicated. Hence, an all-reduce will be necessary.
  ## 2) k is not used in the reference, and mu_k is mapped.
  ## Equivalently, if k is used, mu_k and pi_k are mapped, they match.
  def is_true_communication (self, ref):
    for idim in self.dims:
      mu_dim = self.map[idim]
      proc_size = 1
      if (mu_dim >= 0):
        proc_size = self.PP.get_dim_size (mu_dim)
      if (proc_size == 1):
        ## Cannot ever have communication if we have only one processor
        ## or rank along the current dimension, even if the pi is mapped.
        continue
      dim_name = self.dims[idim]
      if (ref.is_dim_used (dim_name)):
        pi_dim = ref.get_pi_by_name (dim_name)
        ## Data-space dimension is mapped and dimensions don't match. 
        ## So we have communication along some space dimension.
        if (mu_dim >= 0 and pi_dim >= 0 and mu_dim != pi_dim):
          return True
        ## Data-space dimension is not mapped (replicated) but only
        ## one processor is mapped. So the result will have to be broadcast.
        if (mu_dim >= 0 and pi_dim == -1):
          return True
      elif (mu_dim >= 0):
        ## The iteration space dimension is not used to access the array.
        ## Hence, it will become a reduction. We have already checked that
        ## the number of processors along the iteration space dimension is greater
        ## than 1 (and hence each processor has at least one other partner to
        ## communicate with).
        return True 
    return False


  ## @Statement: Generate outgoing communication for current statement.
  def generate_outgoing_communication (self, df, PP, producers, mrap):
    if (self.is_data_sink ()):
      return
    num_acc = len(self.accs)
    ref = self.accs[num_acc-1]
    red_dim = self.get_mapped_reduction_dimension (ref, PP)
    comm_type = self.determine_communication_type (ref, PP)
    self.indent (df)
    df.write ('// Communication for outgoing-data\n')
    self.indent (df)
    df.write ('// Array {}[], comm.type = {}'.format (ref.get_name (), comm_type_str (comm_type)))
    df.write ('\n')
    self.indent (df)
    df.write ('// mu-vector: ')
    self.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// pi-vector: ')
    ref.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// reduction dimension: {}\n'.format (red_dim))
    df.write ('\n')
    if (self.is_true_communication (ref)):
      self.indent (df)
      ref.generate_outgoing_communication (df, self, comm_type, PP)
      local_indent='  '
      self.stop_communication_timer (df, local_indent)
      mrd = self.get_mapped_reduction_dimension (ref, PP)
      if (option_debug >= 5):
        print ("======================================> MRD = {}".format (mrd))
      has_map_red_dim = self.has_mapped_reduction_dimension (ref, PP)
      if (not self.is_data_generator () and 
        (has_map_red_dim or comm_type == COMM_TYPE_LOCAL_SLICE)): ## Was 'or'
        #interm = self.generate_intermediate_allred_buffer (df, ref)
        interm = self.get_allred_intermediate (ref)
        self.restart_computation_timer (df, local_indent)
        self.add_to_global_output (df, PP, producers, mrap)
        self.stop_computation_timer (df, local_indent)
      if (not self.is_data_generator () and not has_map_red_dim and comm_type == COMM_TYPE_ALLRED):
        interm = self.get_allred_intermediate (ref)
        self.restart_computation_timer (df, local_indent)
        self.add_to_global_output (df, PP, producers, mrap)
        self.stop_computation_timer (df, local_indent)
    self.debug_store_computation_result (df, PP)


  def get_debug_filename (self, ref):
    return '{}_at_{}'.format (ref.get_name (), self.name)

  ## @Statement; store the matrix (or slices of it) that have been computed 
  ## at a compute-statement.
  def debug_store_computation_result (self, df, PP):
    num_acc = len(self.accs)
    if (not self.is_data_sink () and not self.is_data_generator ()):
      array = self.accs[num_acc-1]
      wtf_args = self.generate_write_to_file_arguments (array, PP, True)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      self.writeln (df, '')
      self.indent (df)
      df.write ('#ifdef DIMAGE_DEBUG\n')
      self.indent (df)
      df.write ('// Storing generated tile for debug.\n')
      call = '{}_tile{}D(\"{}\", {});'.format (wtf_func, num_array_dim, self.get_debug_filename (array), wtf_args)
      self.indent (df)
      self.writeln (df, call)
      self.indent (df)
      self.writeln (df, '#endif\n')

  ## Statement.declare_outgoing_slices(): Declare a temporary variable to be 
  ## used for an output all-gather or all-reduce array.
  def declare_outgoing_slices (self, df, PP):
    if (self.is_data_generator ()):
      ref = self.accs[0]
      # All (local) data-slices will be allocated as a list of tiles, where 
      # each tile extent is the array extent divided by the lcm of the 
      # processor grid. The number of tiles along each dimension depends on 
      # the array pi-map.
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      if (option_debug >= 5):
        print ("[INFO@declare_outgoing_slices] Calling get_processor_geometry_list_from_map (ref={}): map={}, False, vec01={}".format (ref.get_name (), self.get_mu_map (), vec01))
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      if (proc_geom == ''):
        print ("stmt {} - Proc-geom is empty".format (self.name))
        sys.exit (42)
      alloc_args = '{}, {}'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (self.is_data_sink ()):
      return
    n_acc = len(self.accs)
    ref = self.accs[n_acc-1]
    comm_type = self.determine_communication_type (ref, PP)
    if (comm_type == COMM_TYPE_GATHER_SLICE):
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* out-slice */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (comm_type == COMM_TYPE_LOCAL_SLICE):
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* ctls */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (comm_type == COMM_TYPE_ALLRED):
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01, ref, self.dims) 
      alloc_args = '{}, {} /* ctar */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    return None

  def generate_intermediate_allred_buffer (self, df, ref):
    interm = self.get_allred_intermediate (ref)
    recv_size = self.get_slice_vol_by_name (ref, PP)
    allocator = DIMAGE_BUFFER_ALLOCATOR
    header_payload = ref.get_aggregated_tile_header_space (None)
    line = '{} * {} = {}({} + {});\n'.format (DIMAGE_DT, interm, allocator, recv_size, header_payload)
    self.indent (df)
    df.write (line)
    return interm

  ## Statement: Deallocate (all-gathered) incoming slice buffers.
  def free_in_slices (self, df, in_slices):
    if (in_slices == None):
      return
    for ref in in_slices:
      self.indent (df)
      self.writeln (df, 'free ({});'.format (ref))

  def free_local_pointers (self, df):
    df.write ('\n')
    for ref in self.accs:
      line = ref.get_free_list ()
      df.write (line)

  # Collect all the induced tile trip counts in a dictionary. 
  # The produced dictionary must be previously initialized to {}.
  # Further, the dictionary must be queried in the lexical order
  # of the iterators as defined by the statement.
  def collect_tile_trip_counts_from_ref (self, ref, tss, producers):
    for dd in self.dims:
      iter_name = self.dims[dd]
      # Get array dimension where iteration is used
      adim = ref.get_dim_if_used (iter_name)
      if (adim >= 0 and not iter_name in tss):
        prod = None
        # Fetch the original producer of the array.
        if (ref.get_name () in producers):
          prod = producers[ref.get_name ()]
        if (prod != None):
          # Fetch the original array in the producer
          prod_ref = prod.get_ref_by_name (ref.get_name ())
          tile_size_str = prod_ref.get_num_proc_along_dim (prod, adim, PP)
          # Uncomment to debug tile-sizes used for each loop @ operator.
          if (DEBUG_BLOCK_SIZE_OP_TEN_DIM):
            print ("At operator {}, reference {}, dimension {}: tile size is {}".format (self.name, ref.get_name (), iter_name, tile_size_str))
          if (tile_size_str != '1'):
            tss[iter_name] = tile_size_str
    return tss

  def collect_tile_trip_counts (self, producers): 
    ret = {}
    for ref in self.accs:
      ret = self.collect_tile_trip_counts_from_ref (ref, ret, producers)
    return ret

  ## Statement.add_to_global_output(): Generate a loop nest to accumulate the 
  ## all-gather result, an intermediate, into the result array.
  def add_to_global_output (self, df, PP, producers, mrap):
    nref = len(self.accs)
    out_ref = self.accs[nref-1]
    df.write ('\n\n')
    self.indent (df)
    df.write ('// Adding local contribution to array {}.\n'.format (out_ref.get_name ()))
    indent = BASE_INDENT
    depth = 1
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, False, producers, True, True)
      if (loop == ''):
        continue
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    ttc = self.collect_tile_trip_counts (producers)
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      if (dim_name in ttc):
        trip = ttc[dim_name]
      loop = self.build_l2_loop (level, trip, True, L2_LOOP_GENMODE_FULL, producers)
      df.write (indent)
      df.write ('//idim = {} - trip = {}\n'.format (dim_name, trip))
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      level += 1
      depth += 1
    # Generate tile fetch code.
    df.write (indent)
    # Fetch tile pointer of outgoing buffer.
    self.generate_tile_fetch_code (df, out_ref, producers, True)
    interm = self.get_allred_intermediate (out_ref)
    df.write (indent)
    # Fetch tile pointer of all-reduced buffer
    self.generate_tile_fetch_code (df, out_ref, producers, False, interm)
    self.insert_omp_pragmas (df, indent)
    # Generate point loops
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, True, producers, True, True)
      if (loop == ''):
        continue
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    stmt_body = self.generate_simple_increment_statement (df, out_ref, PP, producers, mrap, indent)
    df.write (indent)
    df.write (stmt_body)
    df.write ('\n')
    for lev in range(depth-1):
      indent = BASE_INDENT * (depth-1)
      df.write (indent)
      df.write ('}\n')
      depth = depth - 1
    df.write (indent)
    interm = self.get_allred_intermediate (out_ref)
    df.write ('free ({});\n'.format (interm))
    df.write ('\n')

    
  # Statement.generate_operator(): Generate the code associated to a full 
  # distributed operator.
  # Data generators are identified by having the prefix 'gen' to their names.
  # Similarly, data sinks are identified by the prefix 'sink'.
  # For data generators, allocate their local tile.
  def generate_operator (self, df, PP, producers, mrap):
    local_indent = '  '
    if (self.is_data_generator ()):
      df.write ('{} *\n'.format (DIMAGE_DT))
    else:
      df.write ('void\n')
    df.write ('{} () {}\n'.format (self.get_operator_name (), '{'))
    self.indent (df)
    df.write ('// Body of operator {}\n'.format (self.name))
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.allocate_local_tile (df, PP, True, self)
      ref.allocate_tile_map (df, PP)
    if (self.is_data_sink ()):
      self.indent (df)
      self.writeln (df, 'MPI_Barrier (MPI_COMM_WORLD);\n')
    if (self.is_data_sink () and DIMAGE_OPTION_DO_CHECK):
      ref = self.accs[0]
      self.indent (df)
      df.write ('{} * {};\n'.format (DIMAGE_DT, ref.get_name_for_check ()))
      self.indent (df)
      ref.allocate_local_tile (df, PP, False, self)
    df.write ('\n')
    self.indent (df)
    df.write ('int {};\n'.format (COMM_SIZE_VAR))
    self.indent (df)
    df.write ('int {} = 0;\n'.format (DIMAGE_BLOCK_COUNT))
    if (self.is_data_sink ()):
      self.indent (df)
      df.write ('int {} = 0;\n'.format (DIMAGE_REFBLOCK_COUNT))
    self.declare_local_iterators (df)
    self.start_communication_timer (df, local_indent)
    in_slices = self.statement_generate_incoming_communication (df, PP)
    self.stop_communication_timer (df, local_indent)
    out_slice = self.declare_outgoing_slices (df, PP)
    self.build_operator_body (df, PP, producers, mrap)
    df.write ('\n')
    # In general, there's no outgoing communication, since we adopt a 
    # push-model mechanism. Nonetheless, in some cases we may need 
    # to perform a collective communication on the global array to 
    # merge partial results and contributions, such as in summa.
    # We include that case here.
    self.start_communication_timer (df, local_indent)
    self.generate_outgoing_communication (df, PP, producers, mrap)
    ## Generate code to initialize tile-map.
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.generate_tile_map_creation_code (df, PP)
    if (out_slice != None):
      self.indent (df)
      self.writeln (df, 'free ({});'.format (out_slice))
    if (self.is_data_sink () and DIMAGE_OPTION_DO_CHECK):
      self.indent (df)
      ref = self.accs[0]
      self.writeln (df, 'free ({});'.format (ref.get_name_for_check ()))
    if (self.is_data_generator ()):
      self.indent (df)
      self.writeln (df, 'MPI_Barrier (MPI_COMM_WORLD);\n')
      ref = self.accs[0]
      ref.dump_generated_tile (df, PP)
      ref.return_allocated (df)
    self.free_local_pointers (df)
    df.write ('}')
    df.write ('\n')
    # Return the name of the generated array when the operator
    # produces one, i.e., return 'None' when the operator is a sink.
    if (self.is_data_generator ()):
      return self.accs[0].get_name ()
    if (self.is_data_sink ()):
      return None
    return self.accs[len(self.accs)-1].get_name ()

  def generate_cannonical_loop_top (self, df, producers):
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      skip_red_dim = False
      loop = self.build_cannonical_loop (level)
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (line)
      level += 1

  def generate_cannonical_loop_bottom (self, df):
    level = len(self.dims)
    for idim in self.dims:
      dim_name = self.dims[idim]
      line = '  ' * level
      df.write (line + '}\n')
      level -= 1


  def generate_single_node_access (self, ref, PP, producers):
    ret = 'sna_' + ref.get_name ()
    ret += '['
    ret += '{}{}D ('.format (DIMAGE_TILE_POINTER, ref.get_num_dim ())
    ret += self.get_tile_size_str_list (ref, self.PP, producers)
    ret += ', '
    ret += self.get_constant_tile_iterator_list (ref, 1)
    ret += ')]'
    return ret

  def generate_single_node_statement (self, PP, producers):
    ret = ''
    is_write = True
    is_acum = False
    use_full_extent = True
    is_allgat_out_slice = False
    operator = ' += '
    num_acc = len(self.accs)
    PP.set_single_node ()
    gen_accs = []
    for ii,ref in enumerate(self.accs):
      acc = self.generate_access (ref, PP, producers, is_write, is_acum, use_full_extent, is_allgat_out_slice)
      gen_accs.append (acc)
    for ii,ref in enumerate(self.accs):
      acc = None
      acc_name = ''
      if (ii == 0):
        idx = len(self.accs)-1
        acc = gen_accs[idx]
        acc_name = self.accs[idx].get_tile_name ()
      else:
        idx = ii-1
        acc = gen_accs[ii-1]
        acc_name = self.accs[idx].get_tile_name ()
      ret += '{}[{}]'.format (acc_name, acc)
      if (ii < num_acc - 1):
        ret += operator
      operator = ' * '
      is_write = False
    ret += ';\n'
    PP.unset_single_node ()
    return ret


  ## Statement.generate_tile_fetch_for_single_node_access ():
  ## Generate the code necessary to fetch the single tile of a non-distributed array.
  ## This function is meant to be used exclusively for generating reference results.
  def generate_tile_fetch_for_single_node_access (self, df, producers):
    df.write ('  // Fetching SNA-references.\n')
    for ref in self.accs:
      #self.generate_tile_fetch_code (df, ref, producers, False)
      line = '  {} *{} = ({} + {});\n'.format (DIMAGE_DT, ref.get_tile_name (), ref.get_sna_ref_name (), DIMAGE_TILE_HEADER_MACRO)
      df.write (line)

  ## Statement.generate_single_node_operator ():
  ## Generate the corresponding reference code for a single operator of the DAG.
  ## This generator will differentiate between data generators, data sinks and 
  ## pure computational operators.
  def generate_single_node_operator (self, df, PP, producers, mrap):
    local_indent = '  '
    indent = local_indent * (len(self.dims) + 1)
    PP.set_single_node ()
    stmt_body = self.generate_single_node_statement (PP, producers)
    if (not self.is_compute_statement ()):
      PP.set_single_node ()
      stmt_body = self.generate_statement_generic (df, PP, producers, mrap, local_indent)
      PP.unset_single_node ()
      indent = local_indent
    if (self.is_data_generator ()):
      ## Read initial data for tensors of generators.
      PP.set_single_node ()
      ref = self.accs[0]
      arr_name = ref.get_sna_ref_name () 
      decl = '  {} *{};'.format (DIMAGE_DT, arr_name)
      df.write (decl)
      bcount_init = '  {} = 0;'.format (DIMAGE_BLOCK_COUNT)
      df.write (bcount_init)
      ref.allocate_local_tile (df, PP, True, self, True)
      PP.unset_single_node ()
    PP.set_single_node ()
    dimage_call = self.generate_dimage_operator_call (df, PP, producers, mrap, local_indent)
    PP.unset_single_node ()
    if (self.is_compute_statement ()):
      df.write ('{}// {} \n'.format (local_indent, self.name))
      df.write (local_indent + '#ifdef DIMAGE_KERNEL_LOOP\n')
      self.generate_tile_fetch_for_single_node_access (df, producers)
      self.generate_cannonical_loop_top (df, producers)
    if (not self.is_compute_statement ()):
      arr_name = self.get_ref (0).get_name ()
      needle = ', {}'.format (arr_name)
      nail = ', sna_{}'.format (arr_name)
      stmt_body = re.sub (needle, nail, stmt_body)
      if (self.is_data_sink ()):
        out_ref = self.get_ref (0)
        wtf_args = self.generate_single_node_write_to_file_arguments (out_ref, PP)
        num_array_dim = out_ref.get_num_dim ()
        stmt_body = '{}_{}D(\"{}\", {});\n'.format (WRITE_MATRIX_TO_FILE, num_array_dim, out_ref.get_sna_reference_filename (), wtf_args)
    df.write (indent + stmt_body)
    if (self.is_compute_statement ()):
      self.generate_cannonical_loop_bottom (df)
      df.write (local_indent + "#else\n")
      dimage_call = re.sub ('tile_', 'sna_', dimage_call)
      df.write (local_indent + dimage_call + '\n')
      df.write (local_indent + "#endif")
      df.write ('\n')
    PP.unset_single_node ()
    if (self.is_data_generator ()):
      return self.accs[0].get_name ()
    if (self.is_data_sink ()):
      return None
    return self.accs[len(self.accs)-1].get_name ()

  def generated_array_name (self):
    varname = re.sub ('^gen','',self.name)
    return varname

  def insert_operator_call (self, df):
    line = '{}();\n'.format (self.get_operator_name())
    if (self.is_data_generator ()):
      line = self.generated_array_name () + ' = ' + line
    df.write (line)

  # Compute and return the volume of an array slice of @ref
  # at the current statement. The slice is computed from
  # dividing an array extent by the number of processors
  # along the mapped dimension.
  def get_slice_vol_by_name (self, ref, PP):
    vol = ""
    for idim in self.dims:
      dim_name = self.dims[idim]
      ispace_pdim = self.map[idim]
      if (ref.is_dim_used (dim_name)):
        dim_size = ref.get_dim_size_if_used (dim_name)
        denom = 1
        array_pdim = ref.get_proc_map_by_dim_name (dim_name)
        # If the iteration space is unmapped, the current statement requires
        # the full slice. Hence, we don't divide the extent.
        if (array_pdim >= 0 and ispace_pdim >= 0):
          denom = PP.get_dim_size (ispace_pdim)
        if (not vol == ""):
          vol += " * "
        term = '{} ({}, {})'.format (DIMAGE_CEIL, dim_size, denom)
        vol = vol + term
    return vol 
    
  def get_local_communication_timer (self):
    return 'timer_KOMM_local_{}'.format (self.name)

  def get_local_computation_timer (self):
    return 'timer_comp_local_{}'.format (self.name)

  def start_computation_timer (self, df, indent):
    df.write (indent)
    df.write ('{} = -{};\n'.format(self.get_local_computation_timer (), DIMAGE_CLOCK))
    df.write ('\n')

  def restart_computation_timer (self, df, indent):
    df.write (indent)
    df.write ('{} += -{};\n'.format(self.get_local_computation_timer (), DIMAGE_CLOCK))
    df.write ('\n')

  def stop_computation_timer (self, df, indent):
    df.write (indent)
    timer_var = self.get_local_computation_timer ()
    df.write ('{} = {} + {};\n'.format(timer_var, DIMAGE_CLOCK, timer_var))
    df.write ('\n')


  def start_communication_timer (self, df, indent):
    df.write (indent)
    df.write ('{} = -{};\n'.format(DIMAGE_START_TIMER, DIMAGE_CLOCK))
    df.write ('\n')

  def stop_communication_timer (self, df, indent):
    df.write (indent)
    timer_var = self.get_local_communication_timer ()
    df.write ('{} += {} + {};\n'.format(timer_var, DIMAGE_CLOCK, DIMAGE_START_TIMER))
    df.write ('\n')

  def declare_timer (self, df): 
    df.write ('double {} = 0.0;\n'.format(self.get_local_computation_timer ()))
    df.write ('double {} = 0.0;\n'.format(self.get_local_communication_timer ()))
    

#################################
## Start of class FFT


FFT_COMP2BW=20

DIMAGE_OP_TYPE_LINALG=2
DIMAGE_OP_TYPE_FFT=3
DIMAGE_OP_TYPE_LINEARIZER=4

DIM_UNMAPPED = -1
RELS_LIST_SEPARATOR = ','

def gen_full (level):
  ret = []
  for nn in range(0,level):
    ret.append([nn])
  return ret


def init_universe (nitems):
  ret = []
  for nn in range(0,nitems):
    ret.append(nn)
  return ret

def set_ones (nitems):
  ret = []
  for nn in range(0,nitems):
    ret.append (nn)
  return set(ret)

def has_item (ss, ii):
  if (not type (ss) == list):
    return ss == ii
  for ee in ss:
    res = has_item (ee, ii)
    if (res):
      return True
  return False

def flatten (ss, partial):
  if (not type (ss) == list):
    return ss
  ret = partial
  for ee in ss:
    if (type(ee) == int):
      ret.add (ee) 
    else:
      ret.union (flatten (ee, partial))
  return ret

def vector_mapped_dims (v):
  ret = 0
  for vi in v:
    ret += vi
  return ret

def vector_count_zeros (v):
  ret = 0
  for vi in v:
    ret += (1 - vi)
  return ret

def vector_equal_mapped_dims (v, w):
  return vector_mapped_dims (v) == vector_mapped_dims (w)

## Determine if w is a subset of v by
## checking if every w_i = 1 is also v_i = 1.
def vector_is_subset (w, v):
  if (len(w) != len(v)):
    print ('[ERROR] Vector w=[{}] and v=[{}] have different lengths'.format (w, v))
    sys.exit (42)
  if (len(v) >= 4):
    ## Special handing for 4D DFTs and beyond.
    zeros_w = vector_count_zeros (w)
    zeros_v = vector_count_zeros (v)
    if (zeros_w == 1 and zeros_v == 1 and w[0] == v[-1] and w[0] == 0):
      return False
    ret = vector_is_subset (w[0:3], v[0:3])
    print ('Is {} subset of {} ==> {}'.format (w, v, ret))
    return ret
  ## Must always have at least one component equal to 1.
  ones = 0
  for wi in w:
    if (wi == 1):
      ones += 1
  ## 
  checked = 0;
  for ii,wi in enumerate(w):
    if (wi == 1 and v[ii] == 1):
      ones -= 1
  return ones == 0

def vector_is_complement (v, w):
  size_v = len(v)
  size_w = len(w)
  if (size_v != size_w):
    print ('[ERROR] Vectors of different length given. Offending vectors are {} and {}. Aborting ...'.format (v, w))
    sys.exit (42)
  ret = True
  for vi,wi in zip(v,w):
    ret = ret and (vi + wi == 1)
  if (ret):
    return ret
  if (size_v >= 4):
    return vector_is_complement (v[0:3], w[0:3])
  return False

def vector_rotate (v):
  if (len(v) >= 4):
    tail = v[3:]
    rot_head = vector_rotate (v[0:3])
    rot_v = rot_head[:] + tail[:]
    return rot_v
  tail = v[-1]
  heads = v[0:-1]
  rot_v = [tail] + heads
  return rot_v
  

## Determine if w is a rotation of v. Return True if so, and False otherwise.
def vector_is_rotation (w, v):
  rot_v = vector_rotate (v)
  return rot_v == w

def vectors_disjoint (v, w):
  disjoint = 0
  for vi, wi in zip(v,w):
    if (vi + wi <= 1):
      disjoint += 1
  return disjoint == len(v)

def vectors_are_rotation (v,w):
  if (not vector_equal_mapped_dims (v,w)):
    return False
  return True

  
def vectors_are_complement_or_rotations (v, w):
  if (v == w):
    return False
  if (vector_is_complement (v,w)):
    return True
  if (vectors_are_rotation (v,w)):
    return True
  return False


def exclude_subsets (space):
  for vv in space:
    for ww in space:
      if (vector_is_subset (ww, vv)):
        print ('Vector [{}] is subset of [{}]'.format (ww, vv))

def count_rotations (v, w, grid_dim, batch_dim):
  if (v == w):
    return ROT_UNREACH
  if (vector_is_complement (v, w)):
    return DIMAGE_DFT_ROT_COMPLEMENT_COST
  if (vector_is_subset (v, w)): 
    return 2 * len(v)**2
  if (vector_is_subset (w, v)):
    return 2 * len(v)**2
  if (vector_equal_mapped_dims (v, w)):
    dist_dims = vector_mapped_dims (v)
    zz = vector_rotate (v)
    rots = 1
    grid_factor = min(dist_dims, grid_dim - batch_dim)
    MAX_ROT = dist_dims ** 2
    while (zz != w):
      zz = vector_rotate (zz)
      rots += 1
      if (rots > MAX_ROT):
        break
    if (rots == len(v)):
      rots = rots * grid_factor
    return rots
  return ROT_UNREACH

def overlap (s1, s2):
  f1 = flatten (s1, set([]))
  f2 = flatten (s2, set([]))
  print ("f1 = {}".format (f1))
  print ("f2 = {}".format (f2))
  nboth = len(f1) + len(f2)
  f3 = f1.union (f2)
  return len (f3) < nboth


def permutations (level, num, full, current):
  if (level >= num):
    return current
  ret = []
  for cs in current:
    for nn in full:
      if (cs == []):
        ret.append ([nn])
      else:
        if (not has_item (cs, nn[0])):
          print ("cs = {}, nn = {}".format (cs, nn))
          s1 = cs[:] + [nn]
          ret.append (s1[:])
  return permutations (level+1, num, full, ret)

def gendecomps (level, num, full, universe):
  if (level >= num):
    return  
  #print ("level {} = {}".format (level, universe))
  for uu in universe:
    for dd in full:
      temp = uu.difference (dd)
      must_add = not temp in universe
      #print ("Candidate={}, dd={}, universe={}, logic.res={}".format (temp.copy (), dd, universe, must_add))
      if (must_add and len(temp) > 0): 
        #print ("Adding {}...".format (temp.copy ()))
        universe.append (temp.copy ())
  gendecomps (level+1, num, full, universe)

def check_feasibility (PP, AA):
  grid_dim = PP.get_num_dim () * 1.0
  avg_proc_dim = int(round(PP.get_max_procs () ** (1.0/grid_dim)))
  ret = True
  for aa in AA:
    tensor = AA[aa]
    viable = tensor.is_viable (avg_proc_dim)
    print ('Tensor {} viable? {}'.format (tensor.get_name (), viable))
    ret = ret and viable
  print ('\n\nOverall viable? {} (avg.proc_dim={})\n\n'.format (ret, avg_proc_dim))
  if (not ret):
    print ('High potential for unfeasible solution space. Aborting ...')
    sys.exit (42)


def exclude_all_parallel (universe, ndim, gdim):
  exclude = set_ones (ndim)
  ret = []
  # Reminder: Sets contain locally-mapped dimensions.
  for ss in universe:
    if (ss == exclude):
      continue
    # Skip modes where the number of distributed dimensions is greater than the
    # number of grid dimensions.
    if (ndim - len(ss) > gdim):
      continue
    if (ndim == 4):   ## 4D tensor used by DFT
      if (len(ss) == 2 and not 3 in ss):
        continue
      if (len(ss) == 1 and 3 in ss):
        continue
    ret.append (ss)
  return ret


def build_string_from_list (term_list, sep):
  ret = ''
  for tt in term_list:
    if (ret != ''):
      ret += sep
    ret += str(tt)
  return ret

def build_sum_from_list (term_list):
  ret = ''
  for tt in term_list:
    if (ret != ''):
      ret += ' + '
    ret += str(tt)
  return ret

def rotate_distribution (distvec, nrot):
  start = len(distvec) - nrot
  return distvec[start:] + distvec[:nrot]

def build_product_from_list (fact_list):
  ret = ''
  for tt in fact_list:
    if (ret != ''):
      ret += ' * '
    ret += '(' + str(tt) + ')'
  return ret

def build_max_expr (base, var, fact):
  ret = '(({} - {}) + {} * ({}))'.format (base, var, var, fact)
  return ret


def build_sum_from_map (term_map):
  ret = ''
  for tt in term_map:
    if (ret != ''):
      ret += ' + '
    term = term_map[tt]
    ret += str(term)
  return ret


##
## Example inputs: shuffler = ['10','01','00'], start_mode = [0,1,1], end_mode=[1,1,0] 
## Compute vec_diff = end_mode - start_mode = [1,0,-1]
## where the non-zeroes, 1 and -1, tell us the entries to swap.
## Entry with -1 is the value being moved out, while entry with 1 is the entry receiving the value.
def swap_entries (shuffler, start_mode, end_mode):
  vec_diff = []
  for vc1,vc2 in zip(start_mode,end_mode):
    vec_diff.append (vc2-vc1)
  i_from = -2
  i_to = -2 
  moved_from = ''
  moved_to = ''
  count_from = 0
  count_to = 0
  compmap = {}
  compmap['01'] = '10'
  compmap['10'] = '01'
  for ii,vv in enumerate(vec_diff):
    if (vv < 0):
      if (i_to < -1):
        i_to = ii
        moved_to = shuffler[i_to]
      count_to += 1
    if (vv > 0):
      if (i_from < -1):
        i_from = ii
        moved_from = shuffler[i_from]
      count_from += 1
  moves = count_from + count_to
  ret = []
  used_dims = 0
  for ii,vv in enumerate(vec_diff):
    if (count_from == count_to):
      if (ii == i_from):
        ret.append (moved_to)
        moves -= 1
      elif (ii == i_to):
        ret.append (moved_from)
        moves -= 1
      elif (vv == 0):
        ret.append (shuffler[ii])
    elif (count_from > count_to):
      if (vv < 0): ## Only one dim will be distributed in the new parallel layout.
        ret.append ('10')
      else:
        ret.append ('00')
    else:
      if (vv < 0): ## Two dim will be distributed in the new parallel mode.
        ret.append (list(compmap)[used_dims])
        used_dims += 1
      else:
        ret.append ('00')
  return ret


class FFT:
  def __init__ (self, form, PP, NP): #, extents, dim_grid, num_PEs):
    self.name = ""
    self.cof = form
    self.np = NP
    self.PP = PP
    self.dim_grid = PP.get_num_dim ()
    self.num_PEs = PP.get_max_procs ()
    ## Shared fields with Statement / Operator of main DiMage script.
    self.ndim = 0  # Number of DFT tensor dimensions excluding possible batch dimension.
    self.nbatchdim = 0
    self.dims = {}
    self.bdims = {}  # batch dims
    self.nref = 0
    self.refs = {}
    self.accs = [] # same as refs but a list
    self.map = {}
    self.last_writer_map = None
    self.ntpd = []
    self.kernel_name = None
    self.linop = False
    self.optype = DIMAGE_OP_TYPE_LINALG
    self.extents = []
    self.mode_map = {}
    self.succ_map = {}
    self.grid = {}
    self.lin_cost_map = {}
    self.next_fft = None
  
  def init_FFT (self):
    if (self.accs == None or len(self.accs) < 2):
      sys.exit (42)
      return
    for ee in self.accs[0].sizes:
      ext = self.accs[0].sizes[ee]
      self.extents.append (ext)
    self.ndim = len(self.extents)
    self.nbatchdim = self.accs[0].get_num_batch_dim ()
    self.ndim = self.ndim - self.nbatchdim
    self.canvec = self.build_canonical_vector ()
    self.universe = [set(init_universe (self.ndim))]
    self.space = gendecomps (0, self.ndim, self.canvec, self.universe)
    #print ('Universe: {}'.format (self.universe))
    exclude = set_ones (self.ndim)
    #print ('Exclude set: {}'.format (exclude))
    self.universe = exclude_all_parallel (self.universe, self.ndim, self.PP.get_num_dim ())
    #print ('Universe (after): {}'.format (self.universe))


  def get_name (self):
    return self.name

  def is_fft (self):
    return True

  def is_linalg (self):
    return False

  def is_linearizer (self):
    return False

  def is_compute_statement (self):
    return True

  def build_canonical_vector (self):
    ret = gen_full(self.ndim)
    return ret


  def uses_tensor (self, tensor):
    for ten in self.accs:
      if (ten.get_name () == tensor.get_name ()):
        return True
    return False

  ## FFT.operator_init_from_file 
  ## Must fuse with operator_init_from_file from main script.
  def operator_init_from_file (self, ff, header):
    line = header
    line = line.strip ()
    # Expect two or three parts: <name>:<iterators>:<kernel_name>
    parts = line.split (':')
    self.name = parts[0]
    # Expect iterators of dimensions separated by ','
    dimlist = parts[1].split (RELS_LIST_SEPARATOR)
    if (len(parts) == 3):
      self.kernel_name = parts[2]
      if (self.kernel_name == 'fft'):
        self.optype = DIMAGE_OP_TYPE_FFT
      if (self.kernel_name == 'lin'):
        self.optype = DIMAGE_OP_TYPE_LINEARIZER
      print ("Kernel name found: {}".format (self.kernel_name))
    for dd,dname in enumerate(dimlist):
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    line = ff.readline ()
    line = line.strip ()
    self.nref = int(line)
    for aa in range(self.nref):
      ref = Reference (self.cof, self.PP, self.np)
      ref.tensor_init_from_file (ff)
      ref.set_parent_operator (self)
      self.refs[ref.get_name ()] = ref
      self.accs.append (ref)

  ## FFT has batched dim?
  def fft_has_batched_dim (self):
    ret = False
    for ten in self.accs:
      ret = ret or ten.has_batch_dim ()
    return ret

  ## Collect the array names of the current FFT object.
  def collect_arrays (self, arrset):
    for ref in self.accs:
      if (ref.get_name () in arrset):
        continue
      arrset[ref.get_name ()] = ref
    return arrset

  def collect_linearized_tensors (self, arrset):
    self.collect_arrays (arrset)

    
  def writeln(self, mf, line):
    mf.write (line + "\n")

  ## FFT
  ## Method to add a generic pre-assembled constraint to the COF object
  ## and to the formulation file.
  def add_constraint (self, mf, cstr, comment=''):
    self.writeln (mf, 'opt.add ({}) ## {}'.format (cstr, comment))
    self.cof.add_cstr (cstr)

  ## FFT.show_info
  def show_info (self):
    print ()
    print ('FFT.info ():')
    print ('Extents : {} (dims={})'.format (self.extents, self.ndim))
    print ('Universe: {}'.format (self.universe))
    print ('Space   : {}'.format (self.space))
    print ('Canonical vector : {}'.format (self.canvec))
    print ('Is batched? : {}'.format (self.fft_has_batched_dim ()))
    for ref in self.accs:
      ref.show_info ()


  def set_to_vec (self, dimset):
    ret = []
    for dd in list(sorted(self.canvec)):
      val = 0
      #print ('Received: {}, Testing={}'.format (dimset, dd))
      if (dd[0] in dimset):
        val = 1
      ret.append (val)
    return ret

  def set_to_vec_str (self, dimset):
    ret = ''
    vec = self.set_to_vec (dimset)
    for dd in vec:
      if (ret != ''):
        ret += '_'
      ret += '{}'.format (dd)
    return ret

  def test_set_to_vec_str (self):
    for vv in self.universe:
      ss = self.set_to_vec_str (vv)
      print ('In test_set_to_vec_str(): Set={}, vec={}'.format (vv, ss))

  ## FFT.exclude_vec_subsets 
  def exclude_vec_subsets (self):
    print ('Excluding subsets from space: <{}>...'.format (self.universe))
    for vv in self.universe:
      vec_v = self.set_to_vec (vv)
      for ww in self.universe:
        vec_w = self.set_to_vec (ww)
        print ('Testing vectors w=[{}] and v=[{}]'.format (vec_w, vec_v))
        if (vector_is_subset (vec_w, vec_v)):
          print ('--> Vector [{}] is subset of [{}]'.format (vec_w, vec_v))

  ## FFT.exclude_non_rotations
  def exclude_non_rotations (self):
    print ('\n\nChecking rotations in space: <{}>...'.format (self.universe))
    for vv in self.universe:
      vec_v = self.set_to_vec (vv)
      for ww in self.universe:
        vec_w = self.set_to_vec (ww)
        print ('Testing vectors w=[{}] and v=[{}]'.format (vec_w, vec_v))
        if (vector_is_rotation (vec_w, vec_v)):
          print ('--> Vector [{}] is rotation of [{}]'.format (vec_w, vec_v))

  def explore_disjoint_space (self):
    print ('\n\nChecking disjoint space: <{}>...'.format (self.universe))
    for vv in self.universe:
      vec_v = self.set_to_vec (vv)
      for ww in self.universe:
        vec_w = self.set_to_vec (ww)
        print ('Testing vectors v=[{}] and v=[{}]'.format (vec_v, vec_w))
        if (vectors_disjoint (vec_v, vec_w)):
          print ('--> Vectors [{}] and [{}] are disjoint'.format (vec_v, vec_w))

  def explore_orthogonal_space (self):
    print ('\n\nChecking disjoint space: <{}>...'.format (self.universe))
    for vv in self.universe:
      vec_v = self.set_to_vec (vv)
      for ww in self.universe:
        vec_w = self.set_to_vec (ww)
        print ('Testing vectors v=[{}] and v=[{}]'.format (vec_v, vec_w))
        if (vectors_are_complement_or_rotations (vec_v, vec_w)):
          print ('--> Vectors [{}] and [{}] are complementary or rotations'.format (vec_v, vec_w))

  ## U-var: Dimension locally mapped "used" var.
  def get_U_var (self, idim):
    ret = 'U_{}_i{}'.format (self.name, idim)
    return ret

  def get_M_var (self, decomp_set, other):
    boolvec = self.set_to_vec_str (decomp_set)
    ret = 'M_{}_{}D_v_{}'.format (self.name, len(decomp_set), boolvec)
    if (other != None):
      ret = other.get_M_var (decomp_set, None)
    return ret

  def get_rotation_obj_var (self, other):
    varname = 'ROT_{}_sum'.format (self.name)
    if (other != None):
      varname = 'ROT_x_{}_to_{}_sum'.format (self.name, other.get_name ())
    return varname

  def get_outgoing_rotation_obj_var (self, vecmode):
    varname = 'ROT_row_{}_{}_sum'.format (self.name, vecmode)
    return varname

  def get_successor_var (self, src_decomp_set, tgt_decomp_set, other):
    src_bvec = self.set_to_vec_str (src_decomp_set)
    tgt_bvec = self.set_to_vec_str (tgt_decomp_set)
    fft_source = self.name
    if (other != None):
      ## succ_x_ variables used between DFTs.
      fft_source = 'x_{}_to_{}'.format (self.name, other.get_name ())
    ret = 'succ_{}_v_{}_v_{}'.format (fft_source, src_bvec, tgt_bvec)
    return ret

  ## PDMC : Parallel Decomposition Mode Cost
  def get_PDMC_var (self, decomp_set):
    boolvec = self.set_to_vec_str (decomp_set)
    ret = 'PDMC_{}_{}D_v_{}'.format (self.name, len(decomp_set), boolvec)
    return ret

  ## LCC: Parallel Decomposition Local Computation Cost
  def get_LCC_var (self, decomp_set):
    boolvec = self.set_to_vec_str (decomp_set)
    ret = 'LCC_{}_{}D_v_{}'.format (self.name, len(decomp_set), boolvec)
    return ret

  ## INC: Inter-node Communication Cost.
  def get_ICC_var (self, decomp_set):
    boolvec = self.set_to_vec_str (decomp_set)
    ret = 'ICC_{}_{}D_v_{}'.format (self.name, len(decomp_set), boolvec)
    return ret

  def declare_variable (self, mf, varname, decl):
    if (decl == None):
      print ("Exiting")
      sys.exit(42)
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname  
    return decl
    
  def set_lower_bound (self, mf, varname, lb):
    cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_upper_bound (self, mf, varname, ub):
    cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    self.set_bounds (mf, varname, lb, ub)
  

  def declare_U_vars (self, mf, decl, do_init=False):
    for idim in range(self.ndim):
      varname = self.get_U_var (idim)
      decl = self.declare_variable (mf, varname, decl)
      if (do_init):
        self.set_bounds_boolean (mf, varname)
    return decl

  def declare_M_vars (self, mf, decl, do_init=False):
    for dmode in self.universe:
      varname = self.get_M_var (dmode, None)
      decl = self.declare_variable (mf, varname, decl)
      dist_vec = self.set_to_vec (dmode)
      if (do_init):
        self.set_bounds_boolean (mf, varname)
      n_dist_dims = vector_count_zeros (dist_vec)
      n_local_dims = len(self.canvec) - 1
      cstr = '{} == 1'.format (varname)
      ## Force solutions of only certain type: ALL-2D or ALL-1D
      if (self.ndim == 3 and option_dft_only_2D and n_dist_dims == len(self.canvec) - 1):
        print ('FORCE SOL of {} ==> {}'.format (dist_vec, n_dist_dims))
        self.add_constraint (mf, cstr, ' # force solution with num.dist.dims = {}.'.format (n_dist_dims))
      if (self.ndim == 4 and 3 in dmode and option_dft_only_2D and n_dist_dims == 2):
        print ('FORCE SOL of {} ==> {}'.format (dist_vec, n_dist_dims))
        self.add_constraint (mf, cstr, ' # force solution with num.dist.dims = {}.'.format (n_dist_dims))
      if (option_dft_only_1D and n_dist_dims == 1):
        print ('FORCE SOL of {} ==> {}'.format (dist_vec, n_dist_dims))
        self.add_constraint (mf, cstr, ' # force solution with num.dist.dims = {}.'.format (n_dist_dims))
    return decl
     

  ## Introduce a weighted sum constraint to limit
  ## the space of possible decompositions to only those where 
  ## numdim <= \sum_{v} len(v) x M^{len(v)}_{v}
  ## Example for 3D-FFT: 3x{i,j,k} + 2x{i,j} + 2x{i,k} + 2x{j,k} + 1x{i} + 1x{j} + 1x{k} >= 3.
  def set_locally_mapped_constraints (self, mf, decl):
    cstr = ''
    for dmode in self.universe:
      modevar = self.get_M_var (dmode, None)
      if (cstr != ''):
        cstr += ' + '
      n_par_dims = len(dmode)
      cstr += '{} * {}'.format (n_par_dims, modevar)
    cstr = '{} <= {}'.format (self.ndim, cstr)
    self.add_constraint (mf, cstr, 'card-mapped constraint')
    for dmode in self.universe:
      sum_str = ''
      modevar = self.get_M_var (dmode, None)
      n_local_u_dims = len(dmode)
      for idim in range(self.ndim):
        u_var = self.get_U_var (idim)
        u_val = 0
        if (idim in dmode):
          u_val = 1
        cstr = '{} >= {} * {}'.format (u_var, u_val, modevar)
        self.add_constraint (mf, cstr, ' M-to-U constraint {}, {}, {}'.format (modevar, idim, u_val))
        if (sum_str != ''):
          sum_str += ' + '
        sum_str += u_var
      cstr = '{} == {}'.format (sum_str, self.ndim)
      self.add_constraint (mf, cstr, 'locally mapped constraints')
    ## Build an upper-bound constraint for each U var from the sum of M vars
    ## that locally-use the dimension associated to U.
    for idim in range(self.ndim):
      u_var = self.get_U_var (idim)
      mlist = []
      for dmode in self.universe:
        if (idim in dmode):
          modevar = self.get_M_var (dmode, None)
          mlist.append (modevar)
      if (len(mlist) > 0):
        sum_cstr = build_sum_constraint (mlist)
        cstr = '{} <= {}'.format (u_var, sum_cstr)
        self.add_constraint (mf, cstr, 'U-var upper-bounded by sum of M')
      mlist = None
    return decl

  def sets_overlap (self, s1, s2):
    common = s1.intersection (s2)
    res = common != set()
    common = None
    return res


  def num_locally_mapped (self, vec):
    ret = 0
    for cc in vec:
      if (cc == 1):
        ret += 1
    return ret

  def is_fully_mapped (self, vec):
    return len(vec) == self.num_locally_mapped (vec)

  ## FFT.
  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname
     

  ## Create capacity constraints per fft.
  ## The given maximum capacity applies to all operators of the DAG.
  ## The memory needed by a statement results from the sum of all its parts.
  def set_statement_capacity_constraint (self, mf, decl, pnc, maxprocs):
    total_expr = ''
    self.writeln (mf, '## Introduced by fft.set_statement_capacity_constraint')
    total_var = self.get_volume_var ()
    decl = self.declare_variable (mf, total_var, decl)
    nn = len(self.accs)
    rid = 1
    term_list = []
    for ref in self.accs:
      # Use the current volume var as a lower bound of the total volume
      volvar = ref.get_volume_var ()
      decl = ref.define_volume_var (mf, decl, False)
      # Set an 'easy' lower-bound for the req_{stmt} variables
      cstr = '{} * {} >= {}'.format (maxprocs, total_var, volvar)
      self.add_constraint (mf, cstr, '# See set_statement_capacity_constraint ()')
      ## Compensate for potential 2-buffers and MPI internal storage.
      ## Extra space for buffers.
      term_list.append (volvar)
      if (rid < nn):
        total_expr += volvar
      else:
        total_expr += '{} * {}'.format (2, volvar)
      rid += 1
    # Only insert the equality: req_ss = \sum_{aa} req_{ss,aa} if we have
    # two or more arrays used ss.
    total_expr = build_sum_from_list (term_list)
    if (len(self.accs) > 1):
      cstr = '{} == {}'.format (total_var, total_expr)
      self.add_constraint (mf, cstr, ' FFT capacity requirements ')
    if (pnc > 0):
      cstr = '{} <= {}'.format (total_var, pnc)
      self.add_constraint (mf, cstr, ' FFT capacity requirements ')
    return decl
    
  def add_constraints_overlapping_modes (self, mf, decl): 
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      vec1 = self.set_to_vec (dmode1)
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, None)
        vec2 = self.set_to_vec (dmode2)
        if (dmode1 != dmode2 and self.num_locally_mapped(vec1) != self.num_locally_mapped (vec2)):
          cstr = '{} + {} <= 1'.format (modevar1, modevar2)
          self.add_constraint (mf, cstr, "overlapping modes cstr")
    return decl


  def add_non_rotating_constraints (self, mf, decl): 
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      vec1 = self.set_to_vec (dmode1)
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, None)
        vec2 = self.set_to_vec (dmode2)
        if (self.num_locally_mapped (vec1) == self.num_locally_mapped (vec2) and vec1 != vec2 and (not vector_is_rotation (vec2, vec1))):
          cstr = '{} <= 1 - {}'.format (modevar2, modevar1)
          self.add_constraint (mf, cstr, 'non-rotating constraint')
    return decl

  def add_non_subset_constraints (self, mf, decl): 
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      vec1 = self.set_to_vec (dmode1)
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, None)
        vec2 = self.set_to_vec (dmode2)
        if (vec1 != vec2 and not self.is_fully_mapped (vec1) and vector_is_subset (vec2, vec1)):
          cstr = '{} <= 1 - {}'.format (modevar2, modevar1)
          self.add_constraint (mf, cstr, 'non-subset constraint')
    return decl

  def get_source_ground_varname (self, vecmode, other):
    varname = 'GR_row_sum_succ_{}_{}'.format (self.name, vecmode)
    if (other != None):
      varname = 'GR_row_sum_succ_x_{}_{}_{}'.format (self.name, other.get_name (), vecmode)
    return varname

  def get_target_ground_varname (self, vecmode, other):
    varname = 'GR_col_sum_succ_{}_{}'.format (self.name, vecmode)
    if (other != None):
      varname = 'GR_col_sum_succ_x_{}_{}_{}'.format (self.name, other.get_name (), vecmode)
    return varname

  def ground_pi_mapping_by_successor_sum (self, refid, src_succ_sum, tgt_succ_sum, vec, cstr_map, other):
    ref = self.accs[refid]
    cstr_map = ref.bound_pi_mapping_by_parmode_vector (vec, src_succ_sum, tgt_succ_sum, cstr_map, other)
    return cstr_map
    

  def create_successor_sum_on_parallel_mode (self, vecmode, term_list, cstr_list, other):
    if (other != None):
      return
    sum_str = ''
    for tt in term_list:
      if (sum_str != ''):
        sum_str += ' + '
      sum_str += tt
    gr_var = self.get_source_ground_varname (vecmode, other)
    cstr = '{} == {}'.format (gr_var, sum_str)
    cstr_list.append (cstr)
    cstr = '{} >= 0, {} <= 1'.format (gr_var, gr_var)
    cstr_list.append (cstr)

  def sum_par_mode_var (self):
    return 'sum_M_{}'.format (self.name)

  def sum_successors_var (self, other):
    if (other == None):
      return 'sum_succ_all_{}'.format (self.name)
    return 'sum_succ_all_x_{}_to_{}'.format (self.name, other.get_name ())

  ## FFT
  ## Create a constraint summing all the M-variables.
  def create_parallel_mode_count_constraint (self, mf, decl):
    sum_str = ''
    term_list = []
    for dmode in self.universe:
      modevar = self.get_M_var (dmode, None)
      term_list.append (modevar)
      self.mode_map[modevar] = -1
    sum_str = build_sum_constraint (term_list)
    # sum_str = \sum M
    sum_var = self.sum_par_mode_var ()
    decl = self.declare_variable (mf, sum_var, decl)
    cstr = '{} == {}'.format (sum_var, sum_str)
    self.add_constraint (mf, cstr)
    cstr = '{} >= 0'.format (sum_var)
    self.add_constraint (mf, cstr)
    cstr = '{} <= {}'.format (sum_var, len(self.universe))
    self.add_constraint (mf, cstr, " sum of M_{} vars ")
    is4d = 0
    if (self.ndim == 4):
      is4d = 1
    if (option_dft_maxdim):
      cstr = '{} == {}'.format (sum_var, len(self.canvec) - is4d)  
      self.add_constraint (mf, cstr, " Force max sum of M_{} ")
    if (option_dft_subdim):
      cstr = '{} == {} - 1'.format (sum_var, len(self.canvec))
      self.add_constraint (mf, cstr, " Force max sum of M_{} to be subdim. ")
    if (option_dft_all_modes):
      cstr = '{} == {}'.format (sum_var, len(self.universe))
      self.add_constraint (mf, cstr, " Force solution to use all M_{} ")
    if (option_dft_n_modes >= 1):
      cstr = '{} == {}'.format (sum_var, option_dft_n_modes)
      self.add_constraint (mf, cstr, " Force solution to have {} rotations ".format (option_dft_n_modes))
    return decl

  def get_mode_num_dist_dims (self, dmode):
    dist_vec = self.set_to_vec (dmode)
    n_dist_dims = vector_count_zeros (dist_vec)
    return n_dist_dims

  def are_modes_1d_and_2d (self, dmode1, dmode2):
    ndd1 = self.get_mode_num_dist_dims (dmode1)
    ndd2 = self.get_mode_num_dist_dims (dmode2)
    return (ndd1 + ndd2 == 3)

    
  ## FFT.build_successor_matrix 
  def build_successor_matrix (self, mf, decl, other = None): 
    if (other != None):
      self.next_fft = other
    pending_cstrs = []
    succ_vars_to_sum = []
    pi_parmode_cstrs = {}
    sum_list = []
    succ_list = {}
    list_sum_vars = []
    allsucc_sum_var = self.sum_successors_var (other)
    decl = self.declare_variable (mf, allsucc_sum_var, decl)
    uni_size = len(self.universe)
    SMrows = {}  # Map to store sum of succ_vars along rows.
    SMcols = {}  # Map to store sum of succ_vars along cols.
    all_inter_dft_succ_list = []
    src_tensor_id = 1
    tgt_tensor_id = 0
    if (other != None):  ## If other is provided, we need to use the output tensor of the current FFT as the source.
      src_tensor_id = 0
      tgt_tensor_id = 1
    bin_var_list = []
    for i1,dmode1 in enumerate(self.universe):
      vec = self.set_to_vec (dmode1)
      vecmode = self.set_to_vec_str (dmode1)
      row_succ_sum_var = self.get_source_ground_varname (vecmode, other)
      col_succ_sum_var = self.get_target_ground_varname (vecmode, other)
      row_rot_sum_var = self.get_outgoing_rotation_obj_var (vecmode)
      list_sum_vars.append (row_succ_sum_var)
      decl = self.declare_variable (mf, row_succ_sum_var, decl)
      decl = self.declare_variable (mf, col_succ_sum_var, decl)
      decl = self.declare_variable (mf, row_rot_sum_var, decl)
      bin_var_list.append (row_succ_sum_var)
      bin_var_list.append (col_succ_sum_var)
      ## Target tensor maps against sum of a row.
      if (other == None and self.next_fft != None and tgt_tensor_id == 0):
        pi_parmode_cstrs = self.ground_pi_mapping_by_successor_sum (tgt_tensor_id, row_succ_sum_var, col_succ_sum_var, vec, pi_parmode_cstrs, other)
      if (other == None and self.next_fft == None and src_tensor_id == 1):
        pi_parmode_cstrs = self.ground_pi_mapping_by_successor_sum (src_tensor_id, col_succ_sum_var[:], row_succ_sum_var, vec, pi_parmode_cstrs, other)
      SMrows[vecmode] = {}
      SMcols[vecmode] = {}
    decl = self.set_variable_bounds_from_list (mf, decl, bin_var_list, 0, 1)
    all_succ_list = []
    SMintermode = {}
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      vec1 = self.set_to_vec (dmode1)
      src_mode_name = self.set_to_vec_str (dmode1)
      term_list = []
      row_rot_sum_list = []
      row_rot_sum_var = self.get_outgoing_rotation_obj_var (src_mode_name)
      print ("Successors: src_mode={}, modevar={}, vec={}".format (dmode1, modevar1, vec1))
      INTER_DFT_CONSTRAINT_PATTERN = '(((1 - {}))) + (({})) + {} + ((((1 - {}))))'
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, other)
        vec2 = self.set_to_vec (dmode2)
        tgt_mode_name = self.set_to_vec_str (dmode2)
        if (dmode1 != dmode2 and not vector_is_subset (vec2, vec1)):  
          succ_var1 = self.get_successor_var (dmode1, dmode2, other)
          decl = self.declare_variable (mf, succ_var1, decl)
          rot_cost = count_rotations (vec1, vec2, self.PP.get_num_dim (), self.nbatchdim)
          ## Uncomment below to print number of rotations between parallel modes.
          ## print ('Rotations between mode {} and mode {} == {}'.format (vec1, vec2, rot_cost))
          if (not vector_is_subset (vec1, vec2)):
            succ_var2 = self.get_successor_var (dmode2, dmode1, other)
            decl = self.declare_variable (mf, succ_var2, decl)
            if (self.are_modes_1d_and_2d (dmode1,dmode2)):
              SMintermode[succ_var1] = succ_var1
              SMintermode[succ_var2] = succ_var2
          if (not vector_is_subset (vec1, vec2)):
            succ_list[succ_var1] = succ_var1
          if (not vector_is_subset (vec2, vec1)):
            succ_list[succ_var2] = succ_var2
          self.succ_map[succ_var1] = -1
          self.succ_map[succ_var2] = -1
          if (not succ_var1 in succ_vars_to_sum and not vector_is_subset (vec2, vec1)):
            succ_vars_to_sum.append (succ_var1 + ' + 0 * 1 ')
          if (other == None and vector_is_complement (vec1, vec2)):
            cstr = '{} + {} <= 1'.format (succ_var1, succ_var2)
            pending_cstrs.append (cstr)
          if (other == None):
            cstr = '{} * {} + {} + {} >= 3 * ({} + {})'.format (modevar1, modevar2, modevar1, modevar2, succ_var1, succ_var2)  # Accelerates solution by 4x.
            pending_cstrs.append (cstr)
          if (other != None):
            ## Connect column successor var of first DFT to row successor var of second DFT.
            dft2_row = other.get_source_ground_varname (tgt_mode_name, None)
            dft2_col = other.get_target_ground_varname (tgt_mode_name, None)
            xyz = self.get_source_ground_varname (src_mode_name, None)
            dft1_col = self.get_target_ground_varname (src_mode_name, None)
            dft1_row = self.get_source_ground_varname (src_mode_name, None)
            cstr = '(((1 - {}))) + (({})) + {} + ((((1 - {})))) >= 4 * {}'.format (dft1_row, dft1_col, dft2_row, dft2_col, succ_var1)  
            self.add_constraint (mf, cstr, ' successor activating constraint. ')
          term_list.append (succ_var1)
          all_succ_list.append (succ_var1)
          all_succ_list.append (succ_var2)
          if (rot_cost < ROT_UNREACH):
            all_inter_dft_succ_list.append ('{} * {}'.format (succ_var1, rot_cost))
            row_rot_sum_list.append ('{} * {}'.format (succ_var1, rot_cost))
          SMrows[tgt_mode_name][succ_var1] = succ_var1
          if (not vector_is_subset (vec1, vec2)):
            SMcols[src_mode_name][succ_var2] = succ_var2
        if (dmode1 == dmode2 and other != None):
          ## Inter-DFT cost expressiono requires allowing for successor variables in the diagonal of the conceptual adjacency matrix.
          succ_var = self.get_successor_var (dmode1, dmode2, other)
          decl = self.declare_variable (mf, succ_var, decl)
          self.set_bounds_boolean (mf, succ_var)
          all_inter_dft_succ_list.append (' - ({})'.format (succ_var))
          dft2_row = other.get_source_ground_varname (tgt_mode_name, None)
          dft2_col = other.get_target_ground_varname (tgt_mode_name, None)
          dft1_col = self.get_target_ground_varname (src_mode_name, None)
          dft1_row = self.get_source_ground_varname (src_mode_name, None)
          # Constraint should be identical to the one inter-DFT case when u != v.
          cstr = '(((1 - {}))) + (({})) + {} + ((((1 - {})))) >= 4 * {}'.format (dft1_row, dft1_col, dft2_row, dft2_col, succ_var)  
          self.add_constraint (mf, cstr, ' Inter-DFT successor activating constraint. ')

      self.create_successor_sum_on_parallel_mode (src_mode_name, term_list, sum_list, other)
      if (other == None): 
        cstr_row_sum_rot = build_sum_from_list (row_rot_sum_list)
        cstr_row_sum_rot = '{} == {}'.format (row_rot_sum_var, cstr_row_sum_rot)
        self.add_constraint (mf, cstr_row_sum_rot, ' row sum rot cstr ' )
        cstr_row_sum_rot = '{} >= {}, {} <= {}'.format (row_rot_sum_var, 0, row_rot_sum_var, uni_size * 10 * DIMAGE_DFT_ROT_COMPLEMENT_COST) ## rotation (all2all) accumulated cost.
        self.add_constraint (mf, cstr_row_sum_rot, ' row sum rot cstr bounds ' )
    ## The sum of successors must amount to at least 1.
    if (other != None):
      for parmode in self.universe:
        vec = self.set_to_vec (parmode)
        col = self.set_to_vec_str (parmode)
        self_succ_var = self.get_successor_var (parmode, parmode, other)
        succ_vars_to_sum.append (self_succ_var)
        self.succ_map[self_succ_var] = -1
    if (option_dft_n_modes > 3): ## Enforce connectivity among 1D and 2D distributions in sub-space search.
      succ_constraint_list = list(SMintermode)
      succ_sum_constraint = build_sum_from_list (succ_constraint_list)
      succ_intermode_constraint = '{} >= 1'.format (succ_sum_constraint)
      self.add_constraint (mf, succ_intermode_constraint, ' Intermode (intra-DFT) successor constraint for sub-space search')
    sum_str = build_sum_from_list (succ_vars_to_sum)
    min_succ = 1
    succ_cstr = '{} >= 0 + 0 + {}'.format (sum_str, min_succ)
    self.add_constraint (mf, succ_cstr, ' min connections of successors.')
    ## Set lower (0) and upper (1) bounds for all successor variables.
    decl = self.set_variable_bounds_from_list (mf, decl, all_succ_list, 0, 1)
    ## Build the WEIGHTED SUM of rotation costs.
    ## Only create sum of rotations for inter-DFT communication.
    ## These constraints ARE NOT generated for intra-DFT communication because
    ## these are handled by similar, but per row, constraints.
    if (self.next_fft != None and other != None):
      rot_sum_var = self.get_rotation_obj_var (other)
      decl = self.declare_variable (mf, rot_sum_var, decl)
      if (True): 
        sum_str = build_sum_from_list (all_inter_dft_succ_list)
        cstr = '{} == 1 + {}'.format (rot_sum_var, sum_str)
        self.add_constraint (mf, cstr, ' Inter DFT succ-sum objective')
        cstr = '{} >= 0'.format (rot_sum_var)
        self.add_constraint (mf, cstr, ' Inter DFT succ-sum objective (LB)')
        cstr = '{} <= ({})'.format (rot_sum_var, uni_size)
        self.add_constraint (mf, cstr, ' Inter DFT succ-sum objective (UB)')
    for cc in sum_list:
      self.add_constraint (mf, cc)
    for sv in succ_list:
      cstr = '{} >= 0, {} <= 1'.format (sv, sv)
      self.add_constraint (mf, cstr)
    ## Compute GR_col_* sums
    col_sum_map = {}
    for parmode in self.universe:
      vec = self.set_to_vec (parmode)
      col = self.set_to_vec_str (parmode)
      succs = SMcols[col]
      temp_list = []
      for sv in succs:
        temp_list.append (succs[sv])
      sum_str = build_sum_from_list (temp_list)
      col_succ_sum_var = self.get_target_ground_varname (col, other)
      col_sum_map[col_succ_sum_var] = col_succ_sum_var
      cstr = '{} == {}'.format (col_succ_sum_var, sum_str) 
      self.add_constraint (mf, cstr)
      ## Source tensor maps against sum of a column of the successor matrix.
    for pivar in pi_parmode_cstrs:
      rhs = build_sum_from_list (pi_parmode_cstrs[pivar])
      local_cstr = '{} >= {}'.format (pivar, rhs) 
      self.add_constraint (mf, local_cstr, ' pi to GR var constraints ')
    ## Bound the sum of successors between 1 and \sum M - 1
    if (True): 
      sum_str = build_sum_from_list (list_sum_vars)
      cstr = '{} == {}'.format (allsucc_sum_var, sum_str)
      self.add_constraint (mf, cstr)
      sum_str = build_sum_from_map (col_sum_map)
      cstr = '{} == {}'.format (allsucc_sum_var, sum_str[:])
      self.add_constraint (mf, cstr)
    cstr = '{} >= 0'.format (allsucc_sum_var)
    self.add_constraint (mf, cstr)
    cstr = '{} >= {}'.format (len(self.universe), self.sum_par_mode_var ())
    self.add_constraint (mf, cstr, ' Bound on sum M')
    cstr = '{} <= {}'.format (allsucc_sum_var, self.sum_par_mode_var ()) 
    self.add_constraint (mf, cstr, " succ sum constraint")
    ## Intra-DFT sum of successors must be sum_M - 1, while inter-DFT sum must be 1.
    if (other == None):
      cstr = '{} == {} - 1'.format (allsucc_sum_var, self.sum_par_mode_var ()) 
      self.add_constraint (mf, cstr, " succ sum == number active modes - 1")
    else:
      cstr = '{} <= 1'.format (allsucc_sum_var)
      self.add_constraint (mf, cstr, " Inter-DFT must be <= 1 ")
    return [pending_cstrs, decl]


  ## FFT
  ## 
  def add_sufficient_work_constraint (self, mf, decl):
    num_procs = self.PP.get_max_procs ()
    if (self.nbatchdim > 0):
      num_procs = self.PP.get_num_procs_per_batch_dim ()
    for i1,pmode in enumerate(self.universe):
      vec_mode = self.set_to_vec (pmode)
      src_mode_name = self.set_to_vec_str (pmode)
      extent_product = self.accs[0].estimate_local_computational_workload (vec_mode)
      proc_weight = self.accs[0].estimate_communication_size (vec_mode)
      cstr = '{} >= ({}) * {}'.format (extent_product, proc_weight, self.get_M_var (pmode, None))
      cstr = '(({}) * ({})) % ({}) == 0'.format (extent_product, self.get_M_var (pmode, None), proc_weight)
      self.add_constraint (mf, cstr, ' sufficient work cstr')
    return decl
    

  def is_any_distributed_str_components (self, vec):
    bdim = self.accs[0].count_batch_dims ()
    for ii in range(len(vec)-bdim):
      if (vec[ii].find ("1") >= 0):
        return True
    return False


  def extract_pi_from_transitions (self, solset, output_tensor = True, prev_state = None):
    self.accs[0].extract_mappings_from_solution_set (solset)
    self.accs[1].extract_mappings_from_solution_set (solset)
    if (not output_tensor):
      return self.accs[0].get_map_key (), None
    D2Gvec = self.accs[0].prep_dist_vec_for_map_key ()
    if (not self.is_any_distributed_str_components (D2Gvec)):
      D2Gvec = self.accs[1].prep_dist_vec_for_map_key ()
      return self.accs[1].get_map_key (), D2Gvec
    if (prev_state != None):
      D2Gvec = prev_state
    nn = len(self.universe)
    Smat = []
    count_s = {}
    count_t = {}
    links = 0
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      Smat.append([])
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, None)
        other = None 
        self_succ_var = self.get_successor_var (dmode1, dmode2, other)
        if (self_succ_var in solset and int(solset[self_succ_var]) == 1):
          count_s[i1] = 1
          count_t[i2] = 1
          Smat[i1].append(1)
          #print (self_succ_var + ' : 1')
          links += 1
        else:
          Smat[i1].append(0)
          #print (self_succ_var + ' : 0')
    #print (Smat)
    #print (count_s)
    #print (count_t)
    next_mode = -1
    mode_list = []
    for k1 in count_s:
      if (not k1 in count_t):
        next_mode = k1
        mode_list.append (next_mode)
        break
    #print ('Start node: {}'.format (next_mode))
    processed = 0
    while processed < links:
      found = -1
      for i1 in range(nn):
        if (Smat[next_mode][i1] == 1):
          next_mode = i1
          found = next_mode
          processed += 1
          mode_list.append (next_mode)
          break
      if (found == -1):
        msg ='[ERROR] Disconnected sequence of parallel modes found.' 
        print (msg)
        gen_error_file (self.name, msg)
        return None, None
    a_list = []
    succ_list = []
    for mm in mode_list:
      parmode = self.universe[mm]
      modevar = self.get_M_var (parmode, None)
      vec = self.set_to_vec (parmode)
      succ_list.append (vec)
      a_list.append (modevar)
    mode_str = build_string_from_list (a_list, ':')
    shuffler = D2Gvec 
    #print ('Shuffler start: {}'.format (shuffler))
    for mm in range(len(succ_list)-1):
      mode_from = succ_list[mm]
      mode_to = succ_list[mm+1]
      #print ('Shuffling entries with states {} ==> {}'.format (mode_from, mode_to))
      shuffler = swap_entries (shuffler, mode_from, mode_to)
      #print ('\t==> Shuffler next: {}  '.format (shuffler))
    #print ('FFT Operator {} (state-list) => {}'.format (self.name, succ_list)) 
    #print ('FFT Operator {} => {}'.format (self.name, mode_str))
    bdim = self.accs[0].count_batch_dims ()
    if (bdim > 0):
      bdim_map = '0' * (self.PP.get_num_dim () - 1) + '1'
      shuffler.append (bdim_map)
    tname = self.accs[0].get_name ()
    if (output_tensor):
      tname = self.accs[1].get_name ()
    ret = tname + '_' + build_string_from_list (shuffler, 'x')
    return ret,shuffler


  def add_constraints_from_array (self, array_cstr, mf, decl):
    for cstr in array_cstr:
      self.add_constraint (mf, cstr)
    return decl

  def add_constraint_fft_faster_than (self, mf, decl, non_ftop):
    cstr = '{} < {}'.format (self.get_gpo_varname (), non_ftop.get_gpo_varname ())
    self.add_constraint (mf, cstr)
    return decl

  ## FFT
  ## Add constraints to guarantee that at least one pi-mapping is >= 1 in the
  ## input and output tensor of the FFT.
  def add_fft_distribution_constraints (self, mf, decl):
    sum_src_pi = self.accs[0].sum_pi_all ()
    sum_tgt_pi = self.accs[1].sum_pi_all ()
    cstr = '{} >= 0'.format (sum_src_pi)
    self.add_constraint (mf, cstr)
    cstr = '{} >= 0'.format (sum_tgt_pi)
    self.add_constraint (mf, cstr)
    ## If the FFT has a batch dimension, fix it to the last dimension of the PE-grid.
    if (self.fft_has_batched_dim ()):
      batch_adim = self.accs[0].ndim - 1
      batch_pdim = self.PP.get_num_dim () - 1 
      cstr = '{} == 1'.format (self.accs[0].get_map_varname (batch_adim, batch_pdim))
      self.add_constraint (mf, cstr)
      cstr = '{} == 1'.format (self.accs[1].get_map_varname (batch_adim, batch_pdim))
      self.add_constraint (mf, cstr)
    return decl

  ## DFT.
  def get_mapped_proc_count (self, dmode):
    fact_list = []
    nbd = self.accs[0].count_batch_dims()
    num_map_dim = 0 
    for idim in range(self.ndim): 
      if (not idim in dmode):
        num_map_dim += 1
    for pp in range (num_map_dim):
      proc_var = self.PP.get_proc_dim_symbol (pp)
      fact_list.append (proc_var)
    if (nbd > 0):
      fact_list.append (self.PP.get_last_dim_var ())
    return build_product_from_list (fact_list)

  def get_unmapped_proc_count (self, dmode):
    fact_list = []
    nbd = self.accs[0].count_batch_dims()
    num_map_dim = 0 
    ngd = self.PP.get_num_dim ()
    hvar_map = {}
    for pp in range(ngd):
      hvar = self.PP.get_proc_dim_symbol (pp)
      hvar_map[hvar] = hvar
    if (nbd > 0):
      del (hvar_map[self.PP.get_last_dim_var ()])
    vec = self.set_to_vec (dmode)
    num_map_dim = vector_count_zeros (vec)
    for pp in range (num_map_dim):
      hvar = self.PP.get_proc_dim_symbol (pp)
      del (hvar_map[hvar])
    if (len (hvar_map) == 0):
      return '(1)'
    return build_product_from_list (hvar_map)

  ## Declare, for each decomposition mode, the cost of the decomposition
  ## together with the local computation cost and the internode communication cost.
  def declare_PDMC_vars (self, mf, decl, do_init=False):
    varlist = []
    constraints = []
    for dmode in self.universe:
      mode_cost = self.get_PDMC_var (dmode)
      decl = self.declare_variable (mf, mode_cost, decl)
      varlist.append (mode_cost)
      local_comp = self.get_LCC_var (dmode)
      decl = self.declare_variable (mf, local_comp, decl)
      varlist.append (local_comp)
      internode_comm = self.get_ICC_var (dmode)
      decl = self.declare_variable (mf, internode_comm, decl)
      varlist.append (internode_comm)
      cstr = '{} == {} + {} * {}'.format (mode_cost, local_comp, internode_comm, FFT_COMP2BW)
      constraints.append (cstr)
      FULL_TENSOR_VOL = self.accs[0].get_full_tensor_volume ()
      mapped_procs = self.get_mapped_proc_count (dmode) 
      ## Estimate accessed volume by dividing tensor volume into count of mapped PEs (product of H_pk vars).
      data_slice_factor = self.get_rotation_comm_unit ()
      local_acc = '{} / ({})'.format (data_slice_factor, mapped_procs)
      local_acc = '{} / ({})'.format (FULL_TENSOR_VOL, mapped_procs)  
      cstr = '{} == {}'.format (local_comp, local_acc)  
      constraints.append (cstr)

      cstr = '{} <= {} * {} * {}'.format (mode_cost, self.accs[0].get_full_tensor_volume (), len(self.universe), MEM2COMP_RATIO)  
      constraints.append (cstr)

      BIG_UB = self.accs[0].get_full_tensor_volume ()
      cstr = '{} <= {}'.format (internode_comm, BIG_UB)  
      constraints.append (cstr)

    if (do_init):
      for vv in varlist:
        self.set_lower_bound (mf, vv, 0)
      for cc in constraints:
        self.add_constraint (mf, cc, ' O = C + K * factor')
    return decl

  def fft_tensor_communication_var (self):
    return 'fft_{}_IO_comm'.format (self.name)

  ## FFT
  def fft_set_linearizer_communication_constraint (self, mf, decl):
    if (len(self.accs) != 2):
      print ('[ERROR] FFT Linearizer operator must access exactly two references.')
      sys.exit (42)
    tensorIn = self.accs[0]
    tensorOut = self.accs[1]
    gpo_var = self.fft_tensor_communication_var ()
    fft_dims = self.dims
    decl = self.declare_variable (mf, gpo_var, decl)
    decl = tensorIn.set_linearizer_communication_constraint (mf, decl, tensorOut, fft_dims, self)
    lcvar_main = tensorIn.linearizer_cost_var (tensorOut)
    cstr = '{} == {}'.format (gpo_var, lcvar_main)
    self.add_constraint (mf, cstr)
    return decl

  ## FFT
  ## Return name of main FFT objective variable.
  def get_gpo_varname (self):
    return 'FFT_{}_obj'.format (self.name)

  ## FFT
  def declare_objective_function_var (self, mf, varname, decl):
    decl = self.declare_variable (mf, varname, decl)
    self.set_lower_bound (mf, varname, 0)
    return decl

  ## FFT
  ## Build a weighted sum of EXEC_COST x M_{parmode}.
  ## More precisely: \sum_{v} = cost(v) x M_{v}
  def build_fft_objective_function (self, mf, decl):
    cstr = ''
    for dmode in self.universe:
      costvar = self.get_PDMC_var (dmode)
      pmode = self.get_M_var (dmode, None)
      if (cstr != ''):
        cstr += ' + '
      cstr += '{} * {}'.format (costvar, pmode)

    mode_list = []
    for i1,dmode1 in enumerate(self.universe):
      vecmode = self.set_to_vec_str (dmode1)
      row_rot_sum_var = self.get_outgoing_rotation_obj_var (vecmode)
      mode_list.append (row_rot_sum_var)
    mode_expr_cost = build_sum_from_list (mode_list)

    perf_expr = ' + ' + cstr
    objvar = self.get_gpo_varname ()
    decl = self.declare_objective_function_var (mf, objvar, decl)
    ten_io_var = self.fft_tensor_communication_var ()
    tensor_vol = self.accs[0].get_volume_var ()
    rot_obj_var = self.get_rotation_obj_var (None)
    cstr = '{} == {}'.format (objvar, perf_expr)
    mapped_procs = int(self.PP.get_max_procs ()) 
    FULL_TENSOR_VOL = self.accs[0].get_full_tensor_volume ()
    if (self.next_fft != None):
      next_fft_rot_obj = self.get_rotation_obj_var (self.next_fft)
      cstr = '{} == (({} * {}) / ({})) * (({}) + {}) {}'.format (objvar, FULL_TENSOR_VOL, MEM2COMP_RATIO, mapped_procs, mode_expr_cost, next_fft_rot_obj, perf_expr)
    else:
      cstr = '{} == (({} * {}) / ({})) * ({}) {}'.format (objvar, FULL_TENSOR_VOL, MEM2COMP_RATIO, mapped_procs, mode_expr_cost, perf_expr) 
    self.add_constraint (mf, cstr, 'Main fft objective function')
    ## Upper-bound for FFT:
    BIG_UB = self.accs[0].get_full_tensor_volume () * 2 * MEM2COMP_RATIO
    if (self.next_fft != None):
      BIG_UB = '2 * ({})'.format (BIG_UB)
    cstr = '{} <= {}'.format (objvar, BIG_UB)
    self.add_constraint (mf, cstr, 'Upper-bound to fft objective function.')
    dft_lb = '(({} * {}) / {})'.format (self.accs[0].get_full_tensor_volume (),  MEM2COMP_RATIO, mapped_procs)
    cstr = '{} >= {}'.format (objvar, dft_lb)
    self.add_constraint (mf, cstr, 'Lower-bound of fft objective function.')
    return decl


  def get_proc_size_var (self, pdim):
    varname = 'H_p{}'.format (pdim)
    return varname

  def get_mu_var (self, idim, pdim):
    varname = 'mu_FFT_i{}_p{}'.format (idim, pdim)
    return varname

  ## 
  def get_pi_var (self, idim, pdim):
    varname = 'pi_FFT_i{}_p{}'.format (idim, pdim)
    return varname

  def declare_mu_vars (self, mf, decl, bin_var_list = None):
    for fdim in self.canvec:
      for pdim in range(self.dim_grid):
        varname = self.get_mu_var (fdim[0], pdim)
        if (bin_var_list != None):
          bin_var_list.append (varname)
        decl = self.declare_variable (mf, varname, decl)
    return [bin_var_list, decl]
    
  def declare_pi_vars (self, mf, decl, bin_var_list = None):
    sys.exit (42)
    for fdim in self.canvec:
      for pdim in range(self.dim_grid):
        varname = self.get_pi_var (fdim[0], pdim)
        if (bin_var_list != None):
          bin_var_list.append (varname)
        decl = self.declare_variable (mf, varname, decl)
    return [bin_var_list, decl]


  ## FFT.
  ## Declare pi-mapping variables for the current FFT operator.
  def declare_ref_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Error in dictionary.")
      sys.exit (42)
    for ref in self.accs:
      decl = ref.declare_map_vars (mf, decl)
    return decl

  ## FFT.
  ## Iterate through accessed tensors, and
  ## set sum bounds for each DS-dimension and each PS-dimension affecting
  ## the pi-mappings of the current reference.
  def set_ref_sum_bounds (self, mf, decl, use_lin_form = False):
    if (not self.is_fft ()):
      print ('[ERROR] Function should only be used with FFT')
      sys.exit (42)
    for rr in self.refs:
      ref = self.refs[rr]
      decl = ref.set_dim_sum_bounds (mf, decl)
      decl = ref.set_linearized_proc_sum_bounds (mf, decl)
    return decl

  def set_variable_bounds_from_list (self, mf, decl, bin_var_list, lb, ub):
    for vv in bin_var_list:
      self.set_bounds (mf, vv, lb, ub)
    return decl

  def set_mu_sum_per_fft_dim (self, mf, decl):
    for fdim in self.canvec:
      sum_cstr = ''
      for pdim in range(self.dim_grid):
        varname = self.get_mu_var (fdim[0], pdim)
        if (sum_cstr != ''):
          sum_cstr += ' + '
        sum_cstr += varname
      cstr = '{} == 1'.format (sum_cstr)
      self.add_constraint (mf, cstr)
    return decl

  def get_comm_set_expression (self, dmode1):
    ret = ''
    for pdim in range(self.dim_grid):
      pvar = self.get_proc_size_var (pdim)
      varlist = []
      for fdim in self.canvec:
        muvar = self.get_mu_var (fdim[0], pdim)
        varlist.append (muvar)
      sum_expr = build_sum_from_list (varlist)
      sum_cstr = '{} * ({})'.format (pvar, sum_expr)
      if (ret != ''):
        ret += ' * '
      ret += sum_cstr
    return ret

  def get_rotation_comm_unit (self):
    vol = 1
    extent_size = 1
    FULL_TENSOR_VOL = self.accs[0].get_full_tensor_volume ()
    ## Divide product by total number of PEs. 
    ## Below we will scale up by the number of PEs in the batch dimension.
    vol = FULL_TENSOR_VOL / self.PP.get_max_procs ()
    ret = '(({}) / ({}))'.format (FULL_TENSOR_VOL, self.PP.get_max_procs ())
    return ret
  
  ## FFT
  ## Main communication constraint between parallel modes of FFT.
  def build_intermode_communication_objective_function (self, mf, decl):
    for i1,dmode1 in enumerate(self.universe):
      modevar1 = self.get_M_var (dmode1, None)
      vec1 = self.set_to_vec (dmode1)
      vecstr1 = self.set_to_vec_str (dmode1)
      for i2,dmode2 in enumerate(self.universe):
        modevar2 = self.get_M_var (dmode2, None)
        if (dmode1 != dmode2 and not self.sets_overlap (dmode1, dmode2)):
          IDV = self.get_ICC_var (dmode1)
          ## Fetch number of processes along H_0
          ## Fetch volume of tensor (along single instance of H_1)
          ## Compute expression.
          data_slice_factor = self.get_rotation_comm_unit ()
          num_procs = '1'
          num_procs = self.PP.get_max_procs ()
          if (self.nbatchdim > 0):
            num_procs = self.PP.get_num_procs_per_batch_dim ()

          ## This constraint has the form: ICC >= (\sum \pi^{T}) x (L - P) x V / PEs_per_batch_dim.
          proc_weight = self.accs[0].estimate_communication_size (vec1)

          row_rot_sum_var = self.get_outgoing_rotation_obj_var (vecstr1)

          #cstr = '{} == ({} * ({}) * {} * ({})) / ({})'.format (IDV, row_rot_sum_var, 1, L_minus_P, proc_weight, data_slice_factor)

          tensor_vol = self.accs[0].get_volume_var ()
          mapped_procs = self.get_mapped_proc_count (dmode1) 
          FULL_TENSOR_VOL = self.accs[0].get_full_tensor_volume ()
          cstr = '{} == ({} * ({})) / ({})'.format (IDV, row_rot_sum_var, FULL_TENSOR_VOL, mapped_procs)    
          USE_UNMAPPED_EXPR = False
          if (USE_UNMAPPED_EXPR):
            unmapped_procs = self.get_unmapped_proc_count (dmode1)
            cstr = '{} == ({} * ({}) * ({})) / ({})'.format (IDV, row_rot_sum_var, FULL_TENSOR_VOL, unmapped_procs, self.PP.get_max_procs ())    
          self.add_constraint (mf, cstr, 'DFT ICC objective function')
          # Lower bound for ICC
          cstr = '{} >= ({} * {}) / ({})'.format (IDV, row_rot_sum_var, FULL_TENSOR_VOL, self.PP.get_max_procs ())
          self.add_constraint (mf, cstr, 'DFT ICC lower-bound ')
    return decl


  ## FFT
  def collect_variables (self, sol):
    for pmode in self.mode_map:
      self.mode_map[pmode] = int(sol[pmode])
    for succ in self.succ_map:
      self.succ_map[succ] = int(sol[succ])
    for pp in range(self.PP.get_num_dim ()): 
      self.grid[pp] = solset[self.PP.get_varname (pp)]
    self.next_fft_rots = None
    if (self.next_fft != None):
      self.next_fft_rots = solset[self.get_rotation_obj_var (self.next_fft)]
    sum_rot = 0
    sum_expr = ''
    for pmode in self.universe:
      vecmode = self.set_to_vec_str (pmode)
      ROT_row = self.get_outgoing_rotation_obj_var (vecmode)
      sum_rot += int(solset[ROT_row])
      if (sum_expr != ''):
        sum_expr += ' + '
      sum_expr += str(solset[ROT_row])
    self.rot_cost = '{} = {}'.format (sum_rot, sum_expr)

  def translate_last_rot_to_distribution (self, sol):
    if (self.next_fft == None):
      return ''
    trans_rot = None
    for solvar in sol:
      if (solvar.find("succ_x_")==0 and int(sol[solvar]) == 1):
        trans_rot = solvar
    if (trans_rot == None):
      return ''
    dist_parts = trans_rot.split ('_v_')
    parts = dist_parts[1].split ('_')
    bdim = self.accs[1].count_batch_dims ()
    pdim = self.PP.get_num_dim ()
    fdim = 0
    alist = []
    vecs = [0] * (len(parts)+bdim)
    for ii,pp in enumerate(parts):
      vecs[ii] = 1-int(pp)
    if (bdim > 0):
      vecs[len(parts)] = 1
    for ii,vv in enumerate(vecs):
      onezero = [0] * pdim
      if (vv == 1):
        onezero[fdim] = vv
        fdim += 1
      key01 = build_string_from_list (onezero, '')
      alist.append (key01)
    ret = self.accs[1].get_name () + '_' + build_string_from_list (alist, 'x')
    return ret

  def translate_parallel_mode_to_mapping_str (self, vecs, tensor_id = 0):
    bdim = self.accs[tensor_id].count_batch_dims ()
    pdim = self.PP.get_num_dim ()
    fdim = 0
    alist = []
    for ii,vv in enumerate(vecs):
      onezero = [0] * pdim
      if (vv == 1):
        onezero[fdim] = vv
        fdim += 1
      key01 = build_string_from_list (onezero, '')
      alist.append (key01)
    ret = self.accs[tensor_id].get_name () + '_' + build_string_from_list (alist, 'x')
    return ret

  def get_tensor_id (self, t_name):
    for ii,tt in enumerate(self.accs):
      if (tt.get_name () == t_name):
        return ii
    return -1

  ##
  def describe (self):
    print ('------------------------------------------------------')
    print ('Grid Shape: ')
    for pp in self.grid:
      print ('{} ==> {}'.format (pp, self.grid[pp]))
    print ('FFT: {} (nbr.batch dims={})'.format (self.name, self.nbatchdim))
    print ('Mode Variables: ')
    for pmode in self.mode_map:
      if (self.mode_map[pmode] == 1):
        print ('** {} ==> {}'.format (pmode, self.mode_map[pmode]))
    print ('Successor Variables: ')
    for psucc in self.succ_map:
      if (self.succ_map[psucc] == 1):
        print ('** {} ==> {}'.format (psucc, self.succ_map[psucc]))
    print ('Rotations (weighted sum): {}'.format (self.rot_cost))
    print ('Rotations (next): {}'.format (self.next_fft_rots))

  ## FFT
  def get_grid_dims_as_str (self):
    temp_pp = []
    for pp in sorted(self.grid):
      temp_pp.append (self.grid[pp])
    return build_string_from_list (temp_pp, '_')

  def describe_to_file (self, ff):
    ff.write ('------------------------------------------------------\n')
    ff.write ('Grid Shape: \n')
    for pp in self.grid:
      ff.write ('{} ==> {}\n'.format (pp, self.grid[pp]))
    ff.write ('FFT: {} (nbr.batch dims={})\n'.format (self.name, self.nbatchdim))
    ff.write ('Mode Variables: \n')
    for pmode in self.mode_map:
      if (self.mode_map[pmode] == 1):
        ff.write ('** {} ==> {}\n'.format (pmode, self.mode_map[pmode]))
    ff.write ('Successor Variables: \n')
    for psucc in self.succ_map:
      if (self.succ_map[psucc] == 1):
        ff.write ('** {} ==> {}\n'.format (psucc, self.succ_map[psucc]))
    ff.write ('Rotations (weighted sum): {}\n'.format (self.rot_cost))
    ff.write ('Rotations (next): {}\n'.format (self.next_fft_rots))
    ff.write ('======================================================\n')


## End of class FFT
##################################


## Processor Space class. Store processor geometry.
class Processor:
  def __init__(self, num_dim, max_procs, proc_vector, form):
    self.np = num_dim
    self.dims = [max_procs] * num_dim
    self.max_procs = max_procs
    self.pvec = proc_vector
    self.single_node = False
    if (proc_vector == None):
      self.pvec = {}
      for pp in range(self.np):
        pname = self.get_varname (pp)
        self.pvec[pp] = pname
    self.sizes = {}
    self.cof = form

  def get_grid_description (self):
    return '{}D_{}p'.format (self.np, self.max_procs)

  def get_num_dim (self):
    return self.np

  def get_sizes (self):
    return self.sizes

  def get_grid_dims_as_str (self):
    temp_pp = []
    for pp in sorted(self.sizes):
      temp_pp.append (self.sizes[pp])
    return build_string_from_list (temp_pp, '_')

  def set_single_node (self):
    self.single_node = True

  def unset_single_node (self):
    self.single_node = False

  def gcd(self):
    if (self.np == 2):
      return gcd(self.sizes[0],self.sizes[1])
    if (self.np == 3):
      return gcd(gcd(self.sizes[0],self.sizes[1]),gcd(self.sizes[1],self.sizes[2]))
    if (self.np > 3):
      sys.exit (42)
    return self.sizes[0]

  def product(self,vec):
    ret=1
    for vv in vec:
      ret = ret * vec[vv]
    return ret

  ## Processor.
  ## Return the number of max procs. divided by the size of the last dimension
  ## of the grid.
  ## This function differs the testing of whether the FFT has a batch dimension
  ## to the FFT operator.
  def get_num_procs_per_batch_dim (self):
    last = self.get_num_dim () - 1
    ret = '({} / {})'.format (self.get_max_procs (), self.get_varname (last))
    return ret

  ## Processor.
  ## FFT
  def get_last_dim_var (self):
    last = self.get_num_dim () - 1
    ret = self.get_varname (last)
    return ret


  def lcm(self):
    sys.exit (42)

  def get_max_procs (self):
    return self.max_procs

  def get_varname (self, pdim):
    varname = 'H_p{}'.format (pdim)
    return varname

  def get_value (self, pdim):
    if (pdim > len(self.pvec)):
      return 0
    return self.pvec[pdim]

  def get_proc_dim_symbol (self, pdim):
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      return self.get_value (pdim)
    else:
      return self.get_varname (pdim)

  def get_dim_lb_str (self, pdim, lb):
    varname = self.get_proc_dim_symbol (pdim)
    ret = '{} >= {}'.format (varname, lb)
    return ret

  def get_dim_ub_str (self, pdim, ub):
    varname = self.get_proc_dim_symbol (pdim)
    ret = '{} <= {}'.format (varname, ub)
    return ret

  ## Return a constraint string of the form: \prod p{i} == max_procs
  def get_product_constraint_str (self):
    expr = ''
    for pp in range(self.np):
      varname = self.get_proc_dim_symbol (pp)
      if (pp > 0):
        expr += ' * '
      expr += str(varname)
    cstr = '{} == {}'.format (expr, self.max_procs)
    return cstr


  def get_proc_sum_expr (self):
    expr = ''
    for pp in range(self.np):
      varname = self.get_proc_dim_symbol (pp)
      if (pp > 0):
        expr += ' + '
      expr += str(varname)
    return '(' + expr + ')'


  ## Declare one processor dimension variable in both the
  ## model and shadow files.
  def declare_dimension (self, pp, mf):
    pvar = self.get_varname (pp)
    cmd = "{} = Int('{}')\n".format (pvar, pvar)
    self.cof.add_var (cmd)
    mf.write (cmd)

  ## Processor.add_processor_space_constraints ():
  ## Declare the grid dimension variables 'pX', bound them
  ## between 1 and the max_procs, and add the constraint
  ## that their product should be equal to max_procs.
  def add_processor_space_constraints (self, mf):
    for pp in range(self.np):
      self.declare_dimension (pp, mf)
    for pp in range(self.np):
      ## Force at least 2 ranks per per process-space dimension.
      ## This will avoid degenerated cases of having 1 processor along 
      ## any one processor space.
      cstr = self.get_dim_lb_str (pp, 2)
      self.cof.add_cstr (cstr)
      cmd = 'opt.add ({})\n'.format (cstr)
      mf.write (cmd)
      cstr = self.get_dim_ub_str (pp, self.max_procs)
      self.cof.add_cstr (cstr)
      cmd = 'opt.add ({})\n'.format (cstr)
      mf.write (cmd)
    LB = 2  #int(self.max_procs ** (1.0 / self.np))
    cstr = '{} >= {}'.format (self.get_proc_dim_symbol (0), LB)
    if (not option_grid_nonasc):
      cstr = '{} >= {}'.format (self.get_proc_dim_symbol (self.np-1), LB)
    self.cof.add_cstr (cstr)
    cmd = 'opt.add ({})\n'.format (cstr)
    mf.write (cmd)
    ## Introduce p_{i} >= p_{i+1} constraints. Accelerates convergence.
    for pp in range(self.np-1):
      var_left = self.get_proc_dim_symbol (pp)
      var_right = self.get_proc_dim_symbol (pp+1)
      cstr = '{} >= {}'.format (var_left, var_right)
      if (not option_grid_nonasc):
        cstr = '{} <= {}'.format (var_left, var_right)
        self.cof.add_cstr (cstr)
        cmd = 'opt.add ({})\n'.format (cstr)
        mf.write (cmd)
    cstr = self.get_product_constraint_str ()
    self.cof.add_cstr (cstr)
    cmd = 'opt.add ({})\n'.format (cstr)
    mf.write (cmd)

  ## Return a comma-separated list of '1' with as many 1s as dimensions in the grid.
  def get_single_node_processor_geometry (self):
    ret = ''
    for ii in range(len(self.sizes)):
      if (ii > 0):
        ret += ', '
      ret += '1'
    return ret

    
  ## Processor.get_processor_geometry_list_from_map (): return a list of comma
  ## separated processor dimensions given a computation or data mapping.
  ## @amap can store the mu-mapping of an operator or the pi-mapping of 
  ## tensor @ref.
  ## We iterate through @amap dimensions and add the number of PEs found along
  ## the mapped dimension.
  ## If @ref == None we will use only the mu-mappings.
  ## If so, a dimension we check whether the current dimension is unmapped 
  ## (pp < 0) or if @use_full is True, we add LCM to the list of PE dimensions.
  ## @use_full is only used when @ref == None.
  ## If @ref != None then we require @iters != None.
  ## Finally, if @ref != None, we can fetch the pi values and compare them.
  def get_processor_geometry_list_from_map (self, amap, use_full, vec01 = None, ref = None, iters = None):
    ret = ''
    for dd in amap:
      ## Determine the number of tiles along a given processor grid dimension.
      if ((vec01 != None and len(amap) == len(vec01) and vec01[dd] == 1) or vec01 == None):
        if (ret != ''):
          ret += ', '
        pp = amap[dd]
        ## If we don't receive the ref object, then we only use @amap.
        if (ref == None):
          if (pp < 0 or use_full):
            ret += str(self.lcm ())
          else:
            ret += str(self.lcm () / self.sizes[pp])
        else:
          ## We received a ref object and the iters (array of iterator names)
          ## We need to compare mu and pi mappings.
          iter_name = iters[dd]
          pp_arr_dim = ref.get_pi_by_name (iter_name)
          if (pp_arr_dim < 0 and pp >= 0):
            ret += str(self.lcm ())
          elif (pp_arr_dim == pp and pp >= 0):
            ret += str(self.lcm () / self.sizes[pp])           
          elif (pp_arr_dim == pp and pp < 0):
            ret += str(self.lcm ())
          else: # pp_arr_dim >= 0 and pp < 0 (Cannot need all but have only one piece)
            ret += 'ERROR'
    return ret

  def get_dim_macro_name (self, pdim):
    macroname = 'DIMAGE_P{}'.format (pdim)
    return macroname

  def get_processor_coordinate_str_list (self):
    ret = ''
    for ii,pp in enumerate(self.dims):
      varname = self.get_processor_coordinate_variable (ii)
      if (ii > 0):
        ret += ', '
      ret += varname
    return ret

  def get_array_of_pointer_processor_coordinates (self):
    ret = ''
    for ii,pp in enumerate(self.dims):
      varname = self.get_processor_coordinate_variable (ii)
      if (ii > 0):
        ret += ', '
      ret += '&{}'.format(varname)
    return ret

  def get_processor_coordinate_variable (self, pp):
    varname = 'dimage_p{}'.format (pp)
    return varname

  ## Declare the variables used to keep the logical coordinates of the
  ## processor grid within each rank.
  ## Also declare an array where the pointers to these variables are stored
  ## and then passed together to debugging functions.
  def declare_processor_coordinate_variables (self, df):
    for ii,pp in enumerate(self.dims):
      proc_var = self.get_processor_coordinate_variable (ii)
      df.write ('int {};\n'.format (proc_var))
    pclist = self.get_array_of_pointer_processor_coordinates ()
    df.write ('int *{}[] = {}{}, NULL{};\n'.format (DIMAGE_RANK_ARRAY, '{', pclist, '}'))


  ## @Processor.init_processor_coordinates : 
  ## Call DIMAGE_PROC_COORD_FUNC to convert the process rank to grid coordinates.
  ## The array of grid dimensions must have been populated in the program.
  def init_processor_coordinates (self, df):
    nd = len(self.dims)
    rank = DIMAGE_PROC_RANK
    grid_dims = DIMAGE_GRID_DIMS
    proc_coords = DIMAGE_PROC_COORDS
    df.write ('  ')
    df.write ('{} ({}, {}, {}, {});\n'.format (DIMAGE_PROC_COORD_FUNC, nd, rank, grid_dims, proc_coords))
    ## Create 'dimage_pX' variables. These will be used to 
    ## access loops and data slices.
    for ii,pp in enumerate(self.dims):
      proc_var = self.get_processor_coordinate_variable (ii)
      dimage_proc_func = DIMAGE_PROC_COORD_FUNC
      df.write ('  ')
      df.write ('{} = {}[{}];\n'.format (proc_var, DIMAGE_PROC_COORDS, ii))

  ## @Processor: Return the number of processor (ranks) along the given dimension.
  def get_dim_size (self, pdim):
    if (pdim < 0 or pdim > len(self.sizes)):
      print ("ERROR: Invalid dimension ({}) used to access Processor-grid object.".format (pdim))
      sys.exit (42)
    if (self.single_node):
      return 1
    return self.sizes[pdim]

  def get_max_dim_size (self):
    return max(self.sizes)
    
  def writeln(self, mf, line):
    mf.write(line + "\n")

  # Declare Z3 variable.
  def declare_variable (self, mf, varname):
    cmd = "{} = Int('{}')".format (varname, varname)
    self.writeln (mf, cmd)
    self.cof.add_var (cmd)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def get_product (self):
    ptotal = 1
    for pp in range(self.np):
      ptotal *= self.dims[pp]
    return ptotal

  def maximize_parallelism (self, mf):
    objvar = 'O_par'
    cmd = "{} = Int('{}')".format (objvar, objvar) # Instead of FP also Int
    self.cof.add_var (cmd)
    self.writeln (mf, cmd)
    expr = ""
    for pp in range(len(self.dims)):
      if (not expr == ""):
        expr += " * "
      expr += 'p{}'.format (pp)
    cstr = '{} <= {}'.format (objvar, expr)
    cmd = 'opt.add ({})'.format (cstr)
    self.cof.add_cstr (cstr)
    self.writeln (mf, cmd)
    cmd = 'P_obj = opt.maximize ({})'.format (objvar)
    self.writeln (mf, cmd)

  def maximize_parallelism_old_ (self, mf):
    objvar = 'O_par'
    cmd = "{} = Int('{}')".format (objvar, objvar)
    self.writeln (mf, cmd)
    self.cof.add_var (cmd)
    expr = ""
    for pp in range(len(self.dims)):
      if (not expr == ""):
        expr += " + "
      expr += 'p{}'.format (pp)
    cmd = "opt.add ({} >= {})".format (objvar, expr)
    self.writeln (mf, cmd)
    cmd = 'P_obj = opt.minimize ({})'.format (objvar)
    self.writeln (mf, cmd)

  def declare_dimensions (self, mf):
    ptotal = self.dims[0] 
    prod_str = ""
    for pp in range(self.np):
      if (pp > 0):
        prod_str += " * "
      pname = self.get_varname (pp)
      self.declare_variable (mf, pname)
      self.set_bounds (mf, pname, 1, ptotal)
      prod_str += pname
    cstr = '{} >= {}'.format (ptotal, prod_str)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def show_geometry (self):
    print ("P = {}".format (self.sizes))

  def read_processor_geometry (self, PP, solset):
    for pp in range(self.np):
      name = PP.get_varname (pp) 
      val = solset[name]
      self.sizes[pp] = int(val)

  ## Store the sizes of the processor grid received as input.
  ## This function is meant to be used *only* with fixed-shape grids.
  def set_dim_sizes_from_fixed_grid (self, procvec):
    for ii,pp in enumerate(procvec):
      self.sizes[ii] = int(pp)

  def get_dimage_grid_varname (self):
    varname = DIMAGE_GRID_DIMS
    return varname

  def generate_processor_space_declarations (self, mf):
    dimlist = ""
    for pp in self.sizes:
      dimlist += '{}'.format (self.sizes[pp])
      dimlist += ', '
    dimlist += '0'
    varname = self.get_dimage_grid_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def print_tikz_graph (self, fout, par_x, par_y):
    for pp,dd in enumerate(self.pvec):
      nodename='p{}'.format (pp)
      nodelabel = '{\\large ' + nodename + '}'
      x=par_x
      y=par_y - 3 * pp
      command = '\\node [shape=rectangle,draw=green,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
    return len(self.pvec)

  def is_dimension_mapped (self, idim):
    if (self.map[idim] >= 0):
      return True
    return False


## Global functions/routines used to:
## Read input *.rels files
## Write the Z3 script 
## Read the Z3 solution.


## Read the first line of an operator from a *.rels
## file. Return the read line and a boolean indicating whether 
## the operator is of FFT, LIN (linearizer) or other (LINALG) type.
def read_operator_header_from_rels_file (ff):
  line = ff.readline ()
  line = line.strip ()
  # Expect two or three parts: <name>:<iterators>:<kernel_name>
  parts = line.split (':')
  # Expect iterators of dimensions separated by ','
  optype = DIMAGE_OP_TYPE_LINALG
  if (len(parts) == 3):
    if (parts[2] == 'fft'):
      optype = DIMAGE_OP_TYPE_FFT
    if (parts[2] == 'lin'):
      optype = DIMAGE_OP_TYPE_LINEARIZER
  return [line, optype]



  

  
## Extract the per-node capacity and convert it to number of elements
## depending on the data type.
def read_per_node_capacity (param_arg):
  factor = 1
  arg = re.sub ('-memcap=','', param_arg) 
  if (arg.find("KB") >= 0):
    arg = re.sub ("KB","",arg)
    factor = 1024
  if (arg.find("MB") >= 0):
    arg = re.sub ("MB","",arg)
    factor = 1024*1024
  if (arg.find("GB") >= 0):
    arg = re.sub ("GB","",arg)
    factor = 1024**3
  if (arg.find ("k") >= 0 or arg.find("K") >= 0):
    arg = re.sub ("[Kk]","",arg)
    factor = 1024
  if (arg.find ("M") >= 0 or arg.find("m") >= 0):
    arg = re.sub ("[Mm]","",arg)
    factor = 1024*1024
  if (arg.find ("G") >= 0 or arg.find("g") >= 0):
    arg = re.sub ("[Gg]","",arg)
    factor = 1024**3
  elem_size = 8
  if (DIMAGE_DT == 'complex'):
    elem_size = 16
  if (DIMAGE_DT == 'float'):
    elem_size = 4
  ret = int(arg) * factor / elem_size
  return ret

def read_max_processors (arg):
  if (arg.find ("procs=") < 0):
    print ("Invalid processor geometry")
    sys.exit (42)
  arg = re.sub ('-procs=','',arg)
  parts = arg.split(",")
  npdim = int(re.sub ('[Dd]','',parts[0]))
  pg = int(re.sub ('[Pp]','',parts[1]))
  return (npdim,pg)

def read_grid_shape (arg):
  if (arg.find ("procs=") < 0):
    print ("Invalid processor geometry")
    sys.exit (42)
  arg = re.sub ('-procs=','',arg)
  parts = arg.split(",")
  ret = []
  prod = 1
  for pp in parts:
    procs = int(pp)
    ret.append (procs)
    prod = prod * procs
  print ('Proc. vec : {}'.format (ret))
  return (ret,prod)

def declare_variable (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Int (\'{}\') # declare int variable\n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl

def declare_float (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Real (\'{}\') # declare float variable\n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl


## Write the main communication objective:
##   K_prog = \sum K_i, where K_i is a computation of the program.
def set_main_comm_objective (SS, mf, decl, cof):
  comm_expr = ""
  for sid in SS:
    stmt = SS[sid]
    if (not comm_expr == ""):
      comm_expr += " + "
    comm_expr += stmt.get_comm_var ()
  obj_var = 'K_prog'
  decl = declare_variable (mf, obj_var, decl, cof)
  comm_expr = '{} >= {}'.format (obj_var, comm_expr)
  cmd = 'opt.add ({})\n'.format (comm_expr)
  cof.add_cstr (comm_expr)
  mf.write (cmd)
  cstr = '{} >= 0'.format (obj_var)
  cmd = 'opt.add ({})\n'.format (cstr)
  cof.add_cstr (cstr)
  mf.write (cmd)
  # NOTE: the minimize command is invoked directly in the COF object.
  cmd = 'K_obj = opt.minimize ({})\n'.format (obj_var)
  mf.write (cmd)
  return decl

## Global objective function for FFTs
def dimage_objective_fft (SS, AA, mf, decl, cof):
  obj_var = 'G_fft' 
  decl = declare_variable (mf, obj_var, decl, cof)
  expr = ''
  cstr = '{} >= 0'.format (obj_var)
  cmd = 'opt.add ({}) ## FFT obj \n'.format (cstr)
  mf.write (cmd)
  cof.add_cstr (cstr)
  for sid in SS:
    oper = SS[sid]
    if (oper.is_fft ()):
      if (expr != ''):
        expr += ' + '
      expr += oper.get_gpo_varname ()
  if (expr != ''):
    cstr = '{} == {}'.format (obj_var, expr)
    cmd = 'opt.add ({}) ## FFT obj \n'.format (cstr)
    mf.write (cmd)
    cof.add_cstr (cstr)
  return decl

## Global objective function for linearizer operators.
def dimage_objective_linearizer (SS, AA, mf, decl, cof):
  obj_var = 'G_lin' 
  decl = declare_variable (mf, obj_var, decl, cof)
  cstr = '{} >= 0'.format (obj_var)
  cmd = 'opt.add ({}) ## FFT obj \n'.format (cstr)
  mf.write (cmd)
  cof.add_cstr (cstr)
  expr = ''
  for sid in SS:
    oper = SS[sid]
    if (oper.is_linearizer ()):
      if (expr != ''):
        expr += ' + '
      obj_var = oper.get_linearizer_objective_var ()
      decl = oper.declare_variable (mf, obj_var, decl)
      oper.set_lower_bound (mf, obj_var, 0)
      expr += obj_var
  if (expr != ''):
    cstr = '{} == {}'.format (obj_var, expr)
    cmd = 'opt.add ({}) ## FFT obj \n'.format (cstr)
    mf.write (cmd)
    cof.add_cstr (cstr)
  return decl

## Set the global performance objective.
def set_global_performance_objectve (SS, mf, decl, cof):
  obj_var = 'G_prog'
  decl = declare_variable (mf, obj_var, decl, cof)
  expr = ''
  for sid in SS:
    stmt = SS[sid]
    if (not stmt.is_compute_statement ()):
      continue
    s_gov = stmt.get_gpo_varname () 
    cstr = '{} >= {}'.format (obj_var, s_gov)
    cmd = 'opt.add ({})\n'.format (cstr)
    Cost_Map[s_gov] = -1
    mf.write (cmd)
    cof.add_cstr (cstr)
    if (not stmt.is_fft () and not stmt.is_linearizer () and stmt.PP.get_num_dim () > 2):
      cstr = '{} <= {} * ({})'.format (obj_var, s_gov, stmt.PP.get_proc_sum_expr ()) # May 23.
      cof.add_cstr (cstr)
      cmd = 'opt.add ({}) # G_prog upper bound \n'.format (cstr)
      mf.write (cmd)
    if (expr != ''):
      expr += ' + '
    expr += s_gov
  cstr = '{} == {}'.format (obj_var, expr)
  cmd = 'opt.add ({})\n'.format (cstr)
  mf.write (cmd)
  cof.add_cstr (cstr)

  cmd = '{} = opt.minimize ({})\n'.format (obj_var, obj_var)
  mf.write (cmd)
  return decl

## Read model solution from solution file. 
## Solution file has the same name as the input file, but with the '.rels'
## extension replaced by '.sol'.
def read_solution_from_file (solfile):
  ff = open (solfile, 'r')
  stf = open (solfile + '.stats', 'w')
  ret = {}
  for line in ff.readlines ():
    line = line.strip ()
    line = re.sub ("\(","",line)
    line = re.sub ("\)","",line)
    if (option_debug >= 3):
      print (line)
    if (line.find ("unsat") >= 0):
      ff.close ()
      stf.close ()
      return None
    if (line.find ("unknown") >= 0):
      ff.close ()
      stf.close ()
      return None
    if (line == "sat"):
      continue
    if (line.find ("->") >= 0):
      continue
    parts = line.split (",")
    # If line components are not separated by ',', then search for ':'
    if (len(parts) <= 1):
      parts = line.split (":")
    if (len(parts) <= 1):
      parts = line.split (" ")
    ret[parts[0]] = parts[1] 
  ff.close ()
  stf.close ()
  return ret

def check_solution (solmap, outfile=None):
  if (solmap == None):
    return
  for vname in sorted(solmap):
    val = int(solmap[vname])
    msg = ''
    if (val < 0):
      msg = '[**ERROR**]: Variable {} has invalid value: {}'.format (vname, val)
    if (msg != ''):
      if (outfile == None):
        print (msg)
      else:
        outfile.write (msg + '\n')

def show_solution_from_table (solset):
  for kk in sorted(solset):
    print ("{} : {}".format (kk, solset[kk]))

def compare_costs (solset):
  k_cost = 0
  w_cost = 0
  for kk in sorted(solset):
    if (kk.find ("K_") == 0):
      k_cost += float(solset[kk])
    if (kk.find ("W_") == 0):
      w_cost += float(solset[kk])
  k_cost = k_cost / 10**9
  w_cost = w_cost / 10**9
  print ("Comm. cost (K): {}".format (k_cost))
  print ("Comp. cost (W): {}".format (w_cost))


def compare_costs_stmt (SS, solset):
  for ii in SS:
    stmt = SS[ii]
    k_var = stmt.get_comm_var ()
    w_var = stmt.get_comp_cost_variable ()
    k_cost = int(solset[k_var])
    w_cost = int(solset[w_var])
    ratio = 'inf'
    if (k_cost != 0):
      ratio = w_cost * 1.0/ k_cost
    print ("Statement {} : K={}, W={}, ratio(w/k)={}".format (stmt.get_name (), k_cost, w_cost, ratio))

def store_solution_to_file (solset, solfile):
  ff = open (solfile, "w")
  for kk in sorted(solset):
    ff.write ("{}:{}\n".format (kk, solset[kk]))
  ff.close ()


def embed_presol_constraints (presol, mf, cof):
  for mv in presol:
    if (mv.find ('pi_') == 0 or mv.find('mu_') == 0 or mv == 'G_MM1'):
      val = presol[mv]
      if (int(val) == 1 or mv == 'G_MM1'):
        cstr = '{} == {}'.format (mv, val)
        cmd = 'opt.add ({})\n'.format (cstr)
        mf.write (cmd)
        cof.add_cstr (cstr)

def build_fft_solution_key (AA, SS, solset):
  if (solset == None):
    return 'nosol'
  if (len(solset) == 0):
    return 'nosol'
  mid_tensor = None
  first_fft = None
  num_fft = 0
  for sid in SS:
    stmt = SS[sid]
    if (stmt.is_fft ()):
      num_fft += 1
  TxG = {}
  TDV = {}
  TxF = {}
  for sid in SS:
    stmt = SS[sid]
    if (stmt.is_fft ()):
      if ((num_fft == 2 and stmt.next_fft != None) or (num_fft == 1)):
        tt,dv = stmt.extract_pi_from_transitions (solset, False, None)
        TxG[stmt.accs[0].get_name ()] = tt
        TDV[stmt.accs[0].get_name ()] = dv
        TxF[stmt.accs[0].get_name ()] = stmt
        tt,dv = stmt.extract_pi_from_transitions (solset, True, None)
        TxG[stmt.accs[1].get_name ()] = tt
        TDV[stmt.accs[1].get_name ()] = dv
        TxF[stmt.accs[1].get_name ()] = stmt
      elif (num_fft == 2 and stmt.next_fft == None):
        source_tensor = stmt.accs[0].get_name ()
        prev_dv = TDV[source_tensor]
        tt,dv = stmt.extract_pi_from_transitions (solset, True, prev_dv)
        TxG[stmt.accs[1].get_name ()] = tt
        TDV[stmt.accs[1].get_name ()] = dv
        TxF[stmt.accs[1].get_name ()] = stmt
  for sid in SS:
    stmt = SS[sid]
    if (stmt.is_fft () and stmt.next_fft != None):
      mid_tensor = stmt.accs[1]
      first_fft = stmt
  #print ('TxG = {}'.format (TxG))
  binmaps = []
  for aa in sorted(AA):
    ref = AA[aa]
    ref.extract_mappings_from_solution_set (solset)
    the_map = ref.get_map_key ()
    if (aa in TxG):
      tensor_id = TxF[aa].get_tensor_id (aa)
      the_map = TxG[aa]
    binmaps.append (the_map)
  return build_string_from_list (binmaps, '_')


def print_program_tikz_graph (tikzfilename, PP, SS, AA):
  ff = open (tikzfilename, 'w')
  ff.write ('\\documentclass[tikz]{standalone}\n')
  ff.write ('\\begin{document}\n')
  ff.write ('\\begin{tikzpicture}\n')
  ff.write ('\\tikzstyle{arrow} = [thick,->,>=stealth]\n')
  rows = 0
  for ss in SS:
    stmt = SS[ss]
    rows += len(stmt.get_dims())
  ## Print processors dims
  x = 5
  y = - int(rows/2)
  PP.print_tikz_graph (ff, x, y)

  ## Print iteration-space dims
  x = 0
  y = 0
  for ss in sorted(SS):
    stmt = SS[ss]
    y -= stmt.print_tikz_graph (ff, x, y)

  ## Print array-space dims
  x = 10
  y = 0
  for aa in sorted(AA):
    ref = AA[aa]
    y -= ref.print_tikz_graph (ff, x, y)

  ff.write ('\\end{tikzpicture}\n')
  ff.write ('\\end{document}')
  ff.close ()
  os.system ('pdflatex {} > /dev/null'.format (tikzfilename))


def show_help ():
  names = []
  desc = []
  otype = []
  defval = []

  names.append('-procs')
  desc.append('Process geometry tuple (e.g., "2D,4p")')
  otype.append('str')
  defval.append('None')

  names.append('-solve')
  desc.append('Invoke Z3 solver')
  otype.append('int')
  defval.append('[1 (Default), 0 (Reuse last solution)]')

  names.append('-obj')
  desc.append('Objective function used')
  otype.append('str')
  defval.append('["comm+comp" (Default), "comm-only", "calc-node-req"]')

  names.append('-memcap')
  desc.append('Memory cap to use')
  otype.append('str')
  defval.append('["0" (Default, no cap)]')

  names.append('-fast')
  desc.append('Fast convergence')
  otype.append('bool')
  defval.append('["False" (Default, no-fast)]')

  names.append('-max-tries')
  desc.append('Max. tries in fast convergence')
  otype.append('int')
  defval.append('["1" (Default)]')

  names.append('-check')
  desc.append('Every operator compares results against a pre-computed reference.')
  otype.append('bool')
  defval.append('[False (Default, no check)]')

  names.append('-graph')
  desc.append('Generate a graph (in PDF) describing the found mapping.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-debug')
  desc.append('Enable debugging level. Internal use.')
  otype.append('int')
  defval.append('[0 (Default, no debug)]')

  names.append('-dft-1d')
  desc.append('Force solutions using only 1D mappings.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-dft-2d')
  desc.append('Force solutions using only 2D mappings.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-dft-allmodes')
  desc.append('Force solutions using all dimension for mappings.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-dft-maxdim')
  desc.append('Force solutions using (G) mappings, where G is the dimensionality of the grid of PEs.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-dft-subdim')
  desc.append('Force solutions using (G-1) mappings, where G is the dimensionality of the grid of PEs.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-verbose')
  desc.append('Show additional mapping information.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-reference')
  desc.append('Generate single node reference (only matmul-like).')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-help')
  desc.append('Show summary of options (this help).')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-used')
  desc.append('Show summary of used options.')
  otype.append('bool')
  defval.append('[False (Default)]')

  print ('***************************************************')
  print ("Showing help:")
  idx=1
  for nn,tt,vv,dd in zip(names,otype,defval,desc):
    print ('[{}] : {} ({}) {} : {}'.format (idx, nn, tt, vv, dd))
    idx += 1
  print ('***************************************************')


##################################################################################
##
##        Main driver starts here.
##
##################################################################################




## Extract options passed to the script.
option_grid = None
option_solve = '-solve=1'
option_objective = '-obj=comm+comp'
option_memcap = "-memcap=0"
option_infile = None
option_check = False
option_graph = False
option_debug = 0
option_timeout = False
option_verbose = False
option_reference = False
option_help = False
option_used = False
option_include_all = False
option_nocodegen = False
option_fast = False
option_max_tries = DIMAGE_MAX_TRIES
option_rename = False
option_init_sol = False
option_init_sol_val = -1
option_grid_nonasc = True
option_dft_only_1D = False
option_dft_only_2D = False
option_dft_maxdim = False
option_dft_subdim = False
option_dft_all_modes = False
option_dft_prerun = False
option_dft_conv = False
option_dft_n_modes = -1
option_regen_summary = False
for arg_id,dimage_option in enumerate(sys.argv):
  if (arg_id < 1):
    continue
  valid = False
  if (dimage_option.find ('-procs=') == 0):
    option_grid = dimage_option
    valid = True
  if (dimage_option.find ('-solve=') == 0):
    option_solve = dimage_option
    valid = True
  if (dimage_option.find ('-obj=') == 0):
    option_objective = dimage_option
    valid = True
  if (dimage_option.find ('-memcap=') == 0):
    option_memcap = dimage_option
    valid = True
  if (dimage_option.find ('-check') == 0):
    option_check = True
    valid = True
  if (dimage_option.find ('-regen-sum') == 0):
    valid = True
    option_regen_summary = True
    option_solve = "=0"
  if (dimage_option.find ('-nocodegen') == 0):
    option_nocodegen = True
    valid = True
  if (dimage_option.find ('-grid-nondesc') == 0):
    option_grid_nonasc = False
    valid = True
  if (dimage_option.find ('-fast') == 0):
    option_fast = True
    valid = True
  if (dimage_option.find ('-max-tries=') == 0):
    option_max_tries = int(re.sub('-max-tries=','',dimage_option))
    valid = True
  if (dimage_option.find ('-graph') == 0):
    option_graph = True
    valid = True
  if (dimage_option.find ('-debug=') == 0):
    option_debug = int(re.sub('-debug=','',dimage_option))
    valid = True
  if (dimage_option.find ('-verbose') == 0):
    option_verbose = True
    valid = True
  if (dimage_option.find ('-reference') == 0):
    option_reference = True
    valid = True
  if (dimage_option.find ('-help') >= 0):
    option_help = True
    valid = True
  if (dimage_option.find ('-used') == 0):
    option_used = True
    valid = True
  if (dimage_option.find ('-timeout=') == 0):
    option_timeout = True
    DIMAGE_TIMEOUT = int(re.sub('-timeout=','',dimage_option))
    valid = True
  if (dimage_option.find ('-rename') == 0):
    option_rename = True
    valid = True
  if (dimage_option.find ('-init-sol=') == 0):
    arg = re.sub('-init-sol=','',dimage_option)
    if (arg == ''):
      print ('[ERROR] -init-sol flag requested, but initial solution value not provided. Aborting ...')
      sys.exit (42)
    option_init_sol_val = int(arg)
    option_init_sol = True
    valid = True
  if (dimage_option.find ('-dft-1d') == 0):
    option_dft_only_1D = True
    option_dft_maxdim = True
    valid = True
  if (dimage_option.find ('-dft-2d') == 0):
    option_dft_only_2D = True
    option_dft_maxdim = True
    valid = True
  if (dimage_option.find ('-dft-maxdim') == 0):
    option_dft_maxdim = True
    valid = True
  if (dimage_option.find ('-dft-subdim') == 0):
    option_dft_subdim = True
    valid = True
  if (dimage_option.find ('-dft-allmodes') == 0):
    option_dft_all_modes = True
    valid = True
  if (dimage_option.find ('-dft-prerun') == 0):
    option_dft_prerun = True
    valid = True
  if (dimage_option.find ('-dft-conv') == 0):
    option_dft_conv= True
    valid = True
  if (dimage_option.find ('-dft-n-modes=') == 0):
    option_dft_n_modes = int(re.sub('-dft-n-modes=','',dimage_option))
    valid = True
    if (option_dft_n_modes < 1):
      print ('[ERROR] Number of given parallel modes ({}) is illegal.'.format (option_dft_n_modes))
      sys.exit (42)
    

  if (len(re.findall ('\.rels$', dimage_option)) == 1):
    option_infile = dimage_option
    valid = True

  if (not valid):
    print ('[ERROR] Option given ({}) is not valid. Aborting ...'.format (dimage_option))
    sys.exit (42)


if (option_fast or option_dft_conv):
  DIMAGE_FAST_SOL = True

if (option_help):
  show_help ()
  sys.exit (0)

if (len(sys.argv) < 3):
  print ("Usage: python dimage.py input.rels -procs=#D,#p [-memcap=C[M|K]] [-solve=0:No|1:Yes] [-obj=comm-only|comm+comp|calc-node-req]")
  print ("Legend:")
  print ("Processor space: -procs=#D,#p")
  print ("Per-node capacity (Optional, Default 0): -memcap=C[MB|KB]")
  print ("Request solve (Optional, Default 1): -solve=0|1; 0=No-solve (Reuse previous solution),1=solve")
  print ("Objective-mode (Optional, Default 'comm+comp'): -obj=comm-only|comm+comp|calc-node-req")
  print ("Example: time python {} 2mm-960.rels -memcap=1024K -procs=2D,8p -solve=1 -obj=comm-only".format (DIMAGE_PY_SCRIPT))
  sys.exit(42)

print ("Summary of received options")
print ("Input File: {}".format (option_infile))
print ("Option memory cap: {}".format (option_memcap))
print ("Option grid: {}".format (option_grid))
print ("Option check: {}".format (option_check))
print ("Option solve: {}".format (option_solve))
print ("Option objective: {}".format (option_objective))
print ("Option verbose: {}".format (option_verbose))
print ("Option debug: {}".format (option_debug))
print ("Option graph: {}".format (option_graph))
print ("Option help: {}".format (option_help))
print ("Option used: {}".format (option_used))

option_estimate_per_node_requirement = False

infile  = option_infile
pnc = read_per_node_capacity (option_memcap)

npdim=0
maxprocs=0
procvec = None

# Extract grid information
if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
  procvec, maxprocs = read_grid_shape (option_grid)
  npdim = len(procvec)
else:
  npdim, maxprocs = read_max_processors (option_grid)


per_node_cap_str_arg = option_grid

# Determine whether to call solver or not.
call_solver = True
if (option_solve.find ("=0") >= 0):
  call_solver = False
if (option_solve.find ("=1") >= 0):
  call_solver = True

## Make comm+comp the default objective mode.
obj_mode = DIMAGE_OBJ_COMM_COMP
if (option_objective.find ("=comm+comp") >= 0):
  obj_mode = DIMAGE_OBJ_COMM_COMP
elif (option_objective.find ("=comm-only") >= 0):
  obj_mode = DIMAGE_OBJ_COMM_ONLY
elif (option_objective.find ("=calc-node-req") >= 0):
 option_estimate_per_node_requirement = True 
elif (option_objective != None):
  print ("[ERROR] Unknown objective selected. Expected '=comm-only', '=comm+comp' or '=calc-node-req'")
  sys.exit (42)

ff = open (infile, "r")

modelfile = re.sub ('\.rels','.model.py', infile)
solfile = re.sub ('\.rels','.sol', infile)
sumfile = re.sub ('\.rels','.sum', infile)
tikzfile =  re.sub ('\.rels','.tex', infile)
cfilename = re.sub ('\.rels','.dimage.c', infile)
if (option_rename):
  file_suffix = '{}D_{}p'.format (npdim, maxprocs)
  modelfile = re.sub ('\.rels','-{}.model.py'.format (file_suffix), infile)
  solfile = re.sub ('\.rels','-{}.sol'.format (file_suffix), infile)
  sumfile = re.sub ('\.rels','-{}.sum'.format (file_suffix), infile)
  tikzfile =  re.sub ('\.rels','.tex', infile)
  cfilename = re.sub ('\.rels','.dimage.c', infile)

mf = open (modelfile + '.shadow', "w")
mf.write('from z3 import *\n\n')
mf.write("opt = Then('simplify','ufnia','qfnra').solver ()\n\n")

cmd = '## Per node max. capacity :{}\n'.format (pnc)
mf.write (cmd)

mf.write ('## Num. processor dimension: {}\n'.format (npdim))
mf.write ('## Max. total processors: {}\n'.format (maxprocs))

# Formulation object
form = Comm_Opt_Form (modelfile, procvec)

# Processor space object
PP = Processor (npdim, maxprocs, procvec, form)
PP.add_processor_space_constraints (mf)

NP = PP.get_num_dim ()

nstmt = int (ff.readline())
SS = {}
CG = [] # control graph
AA = {}
for ss in range(nstmt):
  line, optype = read_operator_header_from_rels_file (ff)
  stmt = None
  if (optype == DIMAGE_OP_TYPE_LINALG):
    stmt = Statement (form, PP, NP)
  elif (optype == DIMAGE_OP_TYPE_FFT):
    stmt = FFT (form, PP, NP)
    print ("Building FFT ...")
  elif (optype ==DIMAGE_OP_TYPE_LINEARIZER):
    stmt = Statement (form, PP, NP)
  else:
    print ("[ERROR] Unexpected type of operator. Aborting ...");
    sys.exit (42)
  stmt.operator_init_from_file (ff, line)
  stmt.init_FFT ()
  AA = stmt.collect_arrays (AA)
  ## Enable statement below to show info for each statement.
  SS[stmt.get_name()] = stmt
  CG.append (stmt)
 
# Gather all the arrays in a separate collection.
for name in AA:
  aa = AA[name]


ff.close ()


if (option_estimate_per_node_requirement):
  estimate_per_node_requirement (SS, PP, procvec)
  sys.exit (0)

decl = {}


if (option_dft_prerun):
  presol = read_solution_from_file ('current.sol')
  mf.write ("\n")
  mf.write ("## Embedding pre-computed MatMul solution.\n")
  embed_presol_constraints (presol, mf, form)
  presol = None

## Set parent operators. Need to do this here because space requirements for
## tensors are added early to the formulation.
for op in SS:
  fft = SS[op]
  if (fft.is_fft ()):
    for aa in AA:
      ref = AA[aa]
      if (fft.uses_tensor (ref)):
        ref.set_parent_operator (fft)

mf.write ("\n")
mf.write ("## Define mu-variables for iteration spaces\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  decl = stmt.declare_map_vars (mf, decl)

if (decl == None):
  print ("decl is None at 4182")
  sys.exit (42)


LT = {}
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    stmt.collect_linearized_tensors (LT)

mf.write ("\n")
mf.write ("## Bound IS-dims and P-dims to not allow multiple mappings of any dimension\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  mf.write ("\n")
  mf.write ('## {} - set_proc_sum_bounds\n'.format (stmt.get_name ()))
  decl = stmt.set_proc_sum_bounds (mf, decl)
  ## Each operator, a statement, will try to declare its own mapping variables.
  ## Existence of mapping variables, mus and pis, are checked first in the decl dictionary.
  mf.write ("\n")
  mf.write ('## {} - declare_ref_vars \n'.format (stmt.get_name ()))
  decl = stmt.declare_ref_vars (mf, decl, LT)
  mf.write ("\n")
  mf.write ('## {} - set_ref_sum_bounds \n'.format (stmt.get_name ()))
  decl = stmt.set_ref_sum_bounds (mf, decl, LT)

mf.write ("\n")
mf.write ("## Special Sum Constraints for linearized tensor operators and references.\n")
for ss in SS:
  stmt = SS[ss]
  if (not stmt.is_fft ()):
    continue
  mf.write ("\n")
  mf.write ('## {} - FFT declare_ref_vars \n'.format (stmt.get_name ()))
  decl = stmt.declare_ref_vars (mf, decl)
  mf.write ("\n")
  mf.write ('## {} - FFT set_ref_sum_bounds \n'.format (stmt.get_name ()))
  decl = stmt.set_ref_sum_bounds (mf, decl, stmt.is_fft())

for arrname in AA:
  continue
  mf.write ("\n")
  mf.write ('## {} - declare_ref_vars \n'.format (arrname))
  aa = AA[arrname]
  mf.write ("\n")
  mf.write ('## {} - set_dim_sum_bounds \n'.format (arrname))
  decl = aa.set_dim_sum_bounds (mf, decl)
  mf.write ("\n")
  mf.write ('## {} - set_proc_sum_bounds \n'.format (arrname))
  decl = aa.set_proc_sum_bounds (mf, decl)

mf.write ("\n")
mf.write ("## Define capacity expressions")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  mf.write ("\n")
  decl = stmt.declare_block_variables (mf, decl)


mf.write ("\n")
mf.write ("## Compute communication slice-expressions")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  decl = stmt.set_comm_slice_expressions (mf, decl, LT, pnc)

if (include_capacity_constraints (per_node_cap_str_arg)):
  mf.write ("\n")
  mf.write ("## Set capacity constraints (original K_*?)\n")
  for ss in SS:
    stmt = SS[ss]
    mf.write ("\n")
    decl = stmt.set_statement_capacity_constraint (mf, decl, pnc, maxprocs)


mf.write ("\n")
mf.write ("## Local-Mapping (lambda / \lambda) constraints: link mu and pi\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  decl = stmt.add_matching_constraints (mf, decl, LT)


mf.write ("\n")
mf.write ("## Declaring replication variables for each array\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  decl = stmt.declare_replication_variables (mf, decl, LT)

### print (LT)

mf.write ("\n")
mf.write ("## Bounding replication variables of each array\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  stmt.bound_replication_variables (mf, LT)

mf.write ("\n")
mf.write ("## Replication constraints: linking rho variables of each array\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  stmt.add_replication_constraints (mf, LT)

mf.write ("\n")
mf.write ("## Replication constraints: setting rho = f(pi_k), for each dim k of A \n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  mf.write ("\n")
  stmt.set_array_dim_replication_expression (mf, LT)


mf.write ("\n")
mf.write ("## Communication (K) constraints\n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  decl = stmt.set_comm_constraints (mf, decl, LT)




onlyFFT = {}

for op in SS:
  stmt = SS[op]
  if (stmt.is_fft ()):
    fft = stmt
    onlyFFT[fft.get_name ()] = fft


sortedFFT = []
if (len(onlyFFT) > 1):
  for ff in sorted(onlyFFT):
    sortedFFT.append (onlyFFT[ff])
  for idx,fft in enumerate(sortedFFT):
    if (idx <  len(sortedFFT) - 1):
      other = sortedFFT[idx+1]
      print ('Connecting FFT {} with FFT {}'.format (fft.get_name (), other.get_name ()))
      fft.next_fft = other


for op in SS:
  stmt = SS[op]
  if (stmt.is_fft ()):
    fft = stmt
    fft.show_info ()
    #fft.test_set_to_vec_str ()
    mf.write ("\n")
    mf.write ("## FFT constraints\n")
    decl = fft.declare_U_vars (mf, decl, True)
    decl = fft.declare_M_vars (mf, decl, True)
    decl = fft.set_locally_mapped_constraints (mf, decl)  ## \sum [d . M^d_v ] >= L
    [array_cstr,decl] = fft.build_successor_matrix  (mf, decl)
    decl = fft.add_constraints_from_array (array_cstr, mf, decl)
    decl = fft.declare_PDMC_vars (mf, decl, True)
    fft_mappings = []
    decl = fft.build_intermode_communication_objective_function (mf, decl)
    decl = fft.build_fft_objective_function (mf, decl)


## Create constraints counting the number of parallel modes used for each FFT.
for op in SS:
  fft = SS[op]
  if (not fft.is_fft ()):
    continue
  decl = fft.create_parallel_mode_count_constraint (mf, decl)
  decl = fft.add_fft_distribution_constraints (mf, decl)

for op in SS:
  fft = SS[op]
  if (not fft.is_fft ()):
    continue
  decl = fft.add_sufficient_work_constraint (mf, decl)


if (len(onlyFFT) > 1):
  for idx,fft in enumerate(sortedFFT):
    if (idx <  len(sortedFFT) - 1):
      other = sortedFFT[idx+1]
      print ('Adding succ-var constraints between FFT {} and FFT {}'.format (fft.get_name (), other.get_name ()))
      [array_cstr,decl] = fft.build_successor_matrix  (mf, decl, other)
      decl = fft.add_constraints_from_array (array_cstr, mf, decl)

  


mf.write ("\n")
mf.write ("## Computation (W) cost expressions \n")
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  if (stmt.is_linearizer ()):
    continue
  if (obj_mode == DIMAGE_OBJ_COMM_COMP):
    decl = stmt.set_mu_dimension_sum (mf, decl)
    decl = stmt.set_computation_cost_expression (mf, decl)
  decl = stmt.set_performance_expression_constraints (mf, decl, obj_mode)

for ss in SS:
  stmt = SS[ss]
  if (not stmt.is_linearizer ()):
    continue
  decl = stmt.set_linearizer_communication_constraint (mf, decl, LT)

## Space slicing for accelerating solver.
if (PP.get_num_dim () > 2):
  mf.write ("\n")
  mf.write ("## Constraints cost(FFT_op) < cost(non FFT_op)\n")
  for s1 in SS:
    ftop = SS[s1]
    if (not ftop.is_fft()):
      continue
    for s2 in SS:
      non_ftop = SS[s2]
      if (non_ftop.is_fft ()):
        continue
      if (non_ftop.is_data_generator ()):
        continue
      if (non_ftop.is_data_sink ()):
        continue
      if (non_ftop.is_linearizer ()):
        continue
      decl = ftop.add_constraint_fft_faster_than (mf, decl, non_ftop)

Cost_Map = {}
mf.write ("\n")
mf.write ("## Global Performance Objective (GPO): \sum_S (W_S + alpha * K_S)\n")
decl = set_global_performance_objectve (SS, mf, decl, form)


mf.close ()

print ('***********************************************\n\n')
start_date = dimage_timestamp ()
print ('DiMage-DFT Start time: {}'.format (start_date))
print ()
print ()

opt_val = -1
if (option_init_sol):
  opt_val = option_init_sol_val
solset = None
GLOBAL_SOL_VAR='G_prog'
it = 1
max_tries = option_max_tries
n_fails = 0
g_sols = []
t_sols = []
start_time = timer ()
CONV_STOP = False
criterion = 100.0
while (call_solver and n_fails < max_tries):
  if (CONV_STOP):
    break
  print ("Iteration #{} (#fails={})".format (it, n_fails))
  print_current_time ()
  form.write_formulation (opt_val, n_fails)
  t_it_start = timer ()
  curr_sol = None
  if (call_solver):
    if (option_debug >= 2):
      print ("Using Z3-solver")
      print ("Model file: {}".format (modelfile))
    os.system ('rm -f {}'.format (solfile))
    cmd=''
    if (option_timeout):
      cmd = 'timeout {} python {} | sort > {}'.format (DIMAGE_TIMEOUT, modelfile, solfile)
    else:
      cmd = 'python {} | sort > {}'.format (modelfile, solfile)
    os.system (cmd)
    curr_sol = read_solution_from_file (solfile) 
    if (curr_sol == None or not GLOBAL_SOL_VAR in curr_sol):
      n_fails += 1
      it += 1
      continue
    if (curr_sol != None and GLOBAL_SOL_VAR in curr_sol):
      new_opt_val = int(curr_sol[GLOBAL_SOL_VAR])
      if (opt_val == -1):
        solset = curr_sol
        opt_val = new_opt_val
        g_sols.append (opt_val)
        print ("Found first solution : {:.2E}".format (Decimal(opt_val)))
      elif (new_opt_val < opt_val):
        solset = curr_sol
        prev_opt_val = opt_val
        opt_val = new_opt_val
        g_sols.append (opt_val)
        if (option_debug >= 2):
          print ("Found improved solution : {}".format (opt_val))
        if (option_dft_conv):
          criterion = ((prev_opt_val - new_opt_val) * 1.0) / (prev_opt_val * 1.0)
          print ('IT {} - crit = {}'.format (it, criterion))
          if (criterion < 0.072):
            CONV_STOP = True
      else:
        print ("No new solution found, reducing step to: {} G_prog vs {} {}".format (n_fails + 2, n_fails + 1, opt_val))
        n_fails += 1
      print ("Iteration #{} - Solution found: {:.2E} ({})".format (it, Decimal(opt_val), opt_val))
      print_current_time ()
      it += 1
      print ("------------------------------------------------------------------")
  t_it_stop = timer ()
  t_sols.append (t_it_stop - t_it_start)
  if (DIMAGE_ONE_SHOT):
    print ('[INFO] One shot used. Terminating ...')
    break

stop_time = timer ()
stop_date = dimage_timestamp ()

solution_key = None
if (call_solver):
  solution_key = build_fft_solution_key (AA, SS, solset)

num_sols = len(t_sols)
if (num_sols == 0):
  num_sols = -1
  n_fails = - n_fails
avg_time_sol = (sum(t_sols)) / (num_sols)
avg_time_all = (stop_time - start_time) / (num_sols + n_fails)
print ("[INFO] No. of solutions found: {}".format (num_sols))
print ("[INFO] No. of attempted retries: {}".format (n_fails))
print ("[INFO] Total solver calls: {} ({} sols + {} fails)".format (it, num_sols, n_fails))
print ("[INFO] Time per solution: {}".format (avg_time_sol))
print ("[INFO] Average time per solver call: {}sec".format (avg_time_all))
print ("[INFO] Search Key: {}".format (solution_key))
if (option_dft_conv):
  print ("[INFO] Convergence criteria achieved = {} ({:.3}). ".format (CONV_STOP, criterion))
print_current_time ()

check_solution (solset)

if (solset == None and call_solver):
  print ("No solution found.")
  sys.exit (1)


## If option_regen_sum is set...
if (option_regen_summary):
  print ("[INFO] Reading previous solution from {}".format (solfile))
  solset = read_solution_from_file (solfile) 
  print (solset)

  solution_key = build_fft_solution_key (AA, SS, solset)

if (call_solver or option_regen_summary):
  if (option_debug >= 2):
    print (solset)
  if (not option_regen_summary):
    store_solution_to_file (solset, solfile)

  sf=open(sumfile,'w')

  sf.write ('***************************************************************\n')
  sf.write ("[INFO] Solver timeout (seconds): {}\n".format (DIMAGE_TIMEOUT))
  sf.write ("[INFO] Per-node capacity (elements): {}\n".format (pnc))
  sf.write ("[INFO] Per-node capacity (X-bytes): {}\n".format (option_memcap))
  sf.write ("[INFO] Dimage Data-Type: {}\n".format (DIMAGE_DT))
  sf.write ("[INFO] DFT Convergence (boolean): {}\n".format (option_dft_conv))
  sf.write ("[INFO] Fast Convergence (boolean): {}\n".format (option_fast))
  sf.write ("[INFO] Max. retries (int): {}\n".format (option_max_tries))
  sf.write ("[INFO] Number of solutions found: {}\n".format (num_sols))
  sf.write ('[INFO] User-given number of parallel modes: {}\n'.format (option_dft_n_modes))
  sf.write ('[INFO] Option all-modes: {}\n'.format (option_dft_all_modes))
  sf.write ('[INFO] Option only-2d modes: {}\n'.format (option_dft_only_2D))
  sf.write ('[INFO] Option only-1d modes: {}\n'.format (option_dft_only_1D))
  sf.write ('[INFO] Option sub-dim modes (=G-1): {}\n'.format (option_dft_subdim))
  sf.write ('[INFO] Option max-dim modes (=G): {}\n'.format (option_dft_maxdim))
  sf.write ("[INFO] Number of attempted retries: {}\n".format (n_fails))
  sf.write ("[INFO] Total solver calls: {} ({} sols + {} fails)\n".format (it, num_sols, n_fails))
  sf.write ("[INFO] Time per solution: {}sec\n".format (avg_time_sol))
  sf.write ("[INFO] Average time per solver call: {}sec\n".format (avg_time_all))
  sf.write ("[INFO] Optimal value found (G_prog): {:.2E}\n".format (Decimal(solset['G_prog'])))
  sf.write ("[INFO] Generated between dates: {} <==> {}\n".format (start_date, stop_date))
  sf.write ("[INFO] Summary re-generated: {}\n".format (option_regen_summary))

  print ('\n[INFO] Showing all solutions')
  sf.write ('\n[INFO] Showing all solutions\n')
  for ii,gg in enumerate(g_sols):
    print ("\tSol.{} : {:.2E} ({})".format (ii, Decimal(gg), gg))
    sf.write ("\tSol.{} : {:.2E} ({})\n".format (ii, Decimal(gg), gg))
  print ('\n')
  sf.write ('\n')


  print ('[INFO] Showing all solver times')
  sf.write ('[INFO] Showing all solver times\n')
  for ii,tt in enumerate(t_sols):
    print ("\tTime-to-Sol.{} : {}sec".format (ii, tt))
    sf.write ("\tTime-to-Sol.{} : {}sec\n".format (ii, tt))
  print ('\n')
  sf.write ('\n')

  ## Collect individual costs.
  for gov in Cost_Map:
    val = int(solset[gov])
    if (val >= 0):
      val = float(solset[gov])
      val = val * 100.0 / float(solset['G_prog'])
    Cost_Map[gov] = val

  any_fft = None
  for ops in SS:
    fft = SS[ops]
    if (fft.is_fft ()):
      any_fft = fft
      fft.collect_variables (solset)
      fft.describe ()
      fft.describe_to_file (sf)

  sf.write ('\n')
  sf.write ('Component Cost (%):\n')
  for gov in Cost_Map:
    msg = '* {} : {}% ({:.2E})\n'.format (gov, Cost_Map[gov], Decimal (solset[gov]))
    sf.write (msg)

  print ('\nComponent Cost (%):')
  for gov in Cost_Map:
    msg = '* {} : {}% ({:.2E})'.format (gov, Cost_Map[gov], Decimal (solset[gov]))
    print (msg)

  sf.write ("\n[INFO] Search Key: {}\n".format (any_fft.get_grid_dims_as_str () + '_' + solution_key))
  print ("\n[INFO] Search Key: {}\n".format (any_fft.get_grid_dims_as_str () + '_' + solution_key))

  print ('======================================================')

  sf.write ("\n\n")
  check_solution (solset, sf)
  sf.write ("\n\n")
  sf.close ()


## If the solver was not invoked, read the latest solution.
if (not call_solver):
  print ("[INFO] Reading previous solution from {}".format (solfile))
  solset = read_solution_from_file (solfile) 
  print (solset)

for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  stmt.extract_mappings_from_solution_set (solset)


for ss in SS:
  stmt = SS[ss]
  if (option_verbose):
    stmt.show_maps ()

if (option_graph):
  print_program_tikz_graph (tikzfile, PP, SS, AA)

if (option_debug >= 5):
  for ss in SS:
    stmt = SS[ss]
    stmt.check_capacity_requirement (solset, pnc)

if (option_reference):
  print ("[INFO] Generating serial references ...")
  for ii,ss in enumerate(SS):
    stmt = SS[ss]
    stmt.gencode_matrix_data_generator (ii+1)

if (option_verbose):
  print ("********************************************************************")
  print ("Showing computed mappings:")
  for ss in SS:
    stmt = SS[ss]
    stmt.report_mappings (AA, PP)
  print ("********************************************************************")

# Cases to align: 
# 1) dim(grid) > dim(comp); 
# 2) replication work (mu unmapped) but pi mapped on generators. 
for ss in SS:
  stmt = SS[ss]
  if (stmt.is_fft ()):
    continue
  stmt.statement_align_mu_mappings (PP)

if (option_verbose):
  show_solution_from_table (solset)

## See note at the top of the script.
if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
  PP.read_processor_geometry (PP, solset)
  if (option_verbose or option_debug >= 2):
    PP.show_geometry ()
else:
  PP.set_dim_sizes_from_fixed_grid (procvec)

if (option_nocodegen):
  print ('Skipping codegen...')
  sys.exit (0)

dist = Dist (PP,SS,CG,cfilename)
dist.codegen (sys.argv, avg_time, it, n_fails, solset)
dist.gen_makefile ()
sys.exit (0)
