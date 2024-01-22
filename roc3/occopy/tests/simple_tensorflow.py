#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://casadi.sourceforge.net/users_guide/html/node6.html

from occopy.dynamics import *
from occopy.transcription import *
import tensorflow as tf
import casadi

class TensorFlowEvaluator(casadi.Callback):
  def __init__(self,t_in,t_out,session, opts={}):
    """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
      session: a tensorflow session
    """
    casadi.Callback.__init__(self)
    assert isinstance(t_in,list)
    self.t_in = t_in
    assert isinstance(t_out,list)
    self.t_out = t_out
    self.construct("TensorFlowEvaluator", opts)
    self.session = session
    self.refs = []

  def get_n_in(self): return len(self.t_in)
  def get_n_out(self): return len(self.t_out)

  def get_sparsity_in(self,i):
      return casadi.Sparsity.dense(*self.t_in[i].get_shape().as_list())

  def get_sparsity_out(self,i):
      return casadi.Sparsity.dense(*self.t_out[i].get_shape().as_list())

  def eval(self,arg):
    # Associate each tensorflow input with the numerical argument passed by CasADi
    d = dict((v,arg[i].toarray()) for i,v in enumerate(self.t_in))
    # Evaluate the tensorflow expressions
    ret = self.session.run(self.t_out,feed_dict=d)
    return ret

  # Vanilla tensorflow offers just the reverse mode AD
  def has_reverse(self,nadj): return nadj==1
  def get_reverse(self,nadj,name,inames,onames,opts):
    # Construct tensorflow placeholders for the reverse seeds
    adj_seed = [tf.placeholder(shape=self.sparsity_out(i).shape,dtype=tf.float64) for i in range(self.n_out())]
    # Construct the reverse tensorflow graph through 'gradients'
    grad = tf.gradients(self.t_out, self.t_in,grad_ys=adj_seed)
    # Create another TensorFlowEvaluator object
    callback = TensorFlowEvaluator(self.t_in+adj_seed,grad,self.session)
    # Make sure you keep a reference to it
    self.refs.append(callback)

    # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
    nominal_in = self.mx_in()
    nominal_out = self.mx_out()
    adj_seed = self.mx_out()
    return casadi.Function(name,nominal_in+nominal_out+adj_seed,callback.call(nominal_in+adj_seed),inames,onames)


a = tf.placeholder(shape=(1,1),dtype=tf.float64)

y = tf.matmul(a, a)

with tf.Session() as session:
    f_tf = TensorFlowEvaluator([a], [y], session)

    dm = DynamicalSystem()

    x = dm.add_state('x')
    u = dm.add_control('u', bounds=(-1, 1))
    x.f = u

    ocp = SinglePhaseOCP(dm)
    ocp.add_lagrangian(f_tf(x))

    ocp.add_bc(ocp.x0.x, 1)
    ocp.add_bc(ocp.xf.x, 1)
    ocp.add_bc(ocp.t0, 0)
    ocp.add_bc(ocp.tf, 3)


    tt = TrapezoidalTranscription(30)
    tt.transcript(ocp)

    ig = DiscreteTrajectory()
    ig.t0 = 0
    ig.tf = 3
    ig.x['x'] = lambda t: 1
    ig.u['u'] = lambda t: 0

    sol = tt.solve(ig, solver_options={'linear_solver': 'ma97'})
    
    print(sol.x)
    sol.autoplot()

