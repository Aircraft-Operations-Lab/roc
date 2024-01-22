# -*- coding: utf-8 -*-

from casadi import *
from roc3.occopy.dynamics import *

def d2l(d):
    try:
        return [v for k, v in d.items()]
    except AttributeError:
        return d

def arr_size(arr):
    s = arr.shape
    return s[0]*s[1]

class TrapezoidalTranscription(object):
    def __init__(self, n_nodes):
        self.n = n_nodes
    def transcript(self, ocp):
        self.var_labels = []
        self.ctr_labels = []
        self.ocp = ocp
        dm = ocp.dm
        n = self.n
        tb = MX.sym('t', 2)
        t0 = tb[0]
        tf = tb[1]
        h = (tf - t0)/(n - 1)
        nx = dm.nx
        ny = dm.ny
        nu = dm.nu
        
        X = MX.sym("X", dm.nx*n)
        U = MX.sym("U", dm.nu*(n - 1))
        Y = MX.sym("Y", dm.ny*n)
        P = MX.sym("P", dm.np)
        
        x_names = [k for k, v in dm.x.items()]
        x = [v for k, v in dm.x.items()]
        y = [v for k, v in dm.y.items()]
        u = [v for k, v in dm.u.items()]
        p = [v for k, v in dm.p.items()]
        t = dm.t
        
        xlb = [v.bounds[0] for v in x]*n
        xub = [v.bounds[1] for v in x]*n
        ulb = [v.bounds[0] for v in u]*(n-1)
        uub = [v.bounds[1] for v in u]*(n-1)
        ylb = [v.bounds[0] for v in y]*n
        yub = [v.bounds[1] for v in y]*n
        plb = [v.bounds[0] for v in p]
        pub = [v.bounds[1] for v in p]
        g = []
        lbg = [0.0]*nx*(n-1)
        ubg = [0.0]*nx*(n-1)
        args = [vertcat(*v) for v in [x, y, u, p]] + [t]
        #pv = args[3]
        # Collocation ---------------------------------------
        f = [
            Function('f_{0}'.format(x_names[j]), args, [x[j].f]) 
            for j in range(dm.nx)
            ]
        for k in range(n-1):
            Xk = X[nx*k:nx*(k+1)]
            Xk1 = X[nx*(k+1):nx*(k+2)]
                
            Yk = Y[ny*k:ny*(k+1)]
            Yk1 = Y[ny*(k+1):ny*(k+2)]
                
            Uk = U[nu*k:nu*(k+1)]
                
            tk = t0 + h*k
            tk1 = t0 + h*(k + 1)
            for j in range(dm.nx):           
                f_jk = f[j](Xk, Yk, Uk, P, tk)
                f_jk1 = f[j](Xk1, Yk1, Uk, P, tk1)
                
                lhs = Xk1[j] - Xk[j]
                rhs =  h*0.5*(f_jk1 + f_jk) #h*f_jk
                
                g.append(lhs - rhs)
                self.ctr_labels.append(f"Collocation {x_names[j]} at node {k}")
        # Path constraints
        args_xu = args
        args_x = [vertcat(*v) for v in [x, y, p]] + [t]
        for ctr in self.ocp.dm.constraints:
            if ctr.ctr_type == 'x':
                fname = 'f_ctr_{0}'.format(id(ctr))
                f_ctr = Function(fname, args_x, [ctr.expr])
                for k in range(n):
                    Xk = X[nx*k:nx*(k+1)]
                    Yk = Y[ny*k:ny*(k+1)]
                    tk = t0 + h*k
                    ctr_val = f_ctr(Xk, Yk, P, tk)
                    g.append(ctr_val)
                    lbg.append(ctr.bounds[0])
                    ubg.append(ctr.bounds[1])
            elif ctr.ctr_type == 'xu':
                fname = 'f_ctr_{0}'.format(id(ctr))
                f_ctr = Function(fname, args_xu, [ctr.expr])
                for k in range(n-1):
                    Xk = X[nx*k:nx*(k+1)]
                    Yk = Y[ny*k:ny*(k+1)]
                    #Xk1 = X[nx*(k+1):nx*(k+2)]
                    #Yk1 = Y[ny*(k+1):ny*(k+2)]
                    Uk = U[nu*k:nu*(k+1)]
                    tk = t0 + h*k
                    #tk1 = t0 + h*(k + 1)
                    
                    ctr_val_0 = f_ctr(Xk, Yk, Uk, P, tk)
                    #ctr_val_1 = f_ctr(Xk1, Yk1, Uk, P, tk1)
                    
                    g.append(ctr_val_0)
                    #g.append(ctr_val_1)
                    lbg.append(ctr.bounds[0])
                    #lbg.append(ctr.bounds[0])
                    ubg.append(ctr.bounds[1])
                    #ubg.append(ctr.bounds[1])
                # LAST NODE
                if ctr.bounds[1] - ctr.bounds[0]:  # ONLY FOR INEQUALITY CONSTRAINTS!
                    Xk = X[nx*(k+1):nx*(k+2)]
                    Yk = Y[ny*(k+1):ny*(k+2)]
                    Uk = U[nu*k:nu*(k+1)]
                    tk = t0 + h*(k+1)
                    ctr_val_0 = f_ctr(Xk, Yk, Uk, P, tk)
                    g.append(ctr_val_0)
                    lbg.append(ctr.bounds[0])
                    ubg.append(ctr.bounds[1])
                

        # Boundary constraints -------------------------------
        args_b = []
        args_b += [vertcat(*d2l(ocp.x0))]
        args_b += [vertcat(*d2l(ocp.y0))]
        args_b += [ocp.t0]
        args_b += [vertcat(*d2l(ocp.xf))]
        args_b += [vertcat(*d2l(ocp.yf))]
        args_b += [ocp.tf]
        args_b += [vertcat(*d2l(p))]
        
        X0 = X[:nx]
        Xf = X[nx*(n-1):]
        
        Y0 = Y[:ny]
        Yf = Y[ny*(n-1):]
        
        argvals_b = (X0, Y0, t0, Xf, Yf, tf, P)
        
        for bc in ocp.bc:
            f_bc = Function('bc_{0}'.format(id(bc)), args_b, [bc.expr])
            g.append(f_bc(*argvals_b))
            lbg.append(bc.bounds[0])
            ubg.append(bc.bounds[1])
        
        g.append(tf - t0)
        lbg.append(0)
        ubg.append(inf)
        # Cost Functional --------------------------------------
        J = 0
        
        f_mayer = Function('mayer', args_b, [ocp.mayer])
        J += f_mayer(*argvals_b)
        
        f_lagr = Function('lagr', args, [ocp.lagrangian])
        for k in range(n-1):
            Xk = X[nx*k:nx*(k+1)]
            Xk1 = X[nx*(k+1):nx*(k+2)]
                
            Yk = Y[ny*k:ny*(k+1)]
            Yk1 = Y[ny*(k+1):ny*(k+2)]
                
            Uk = U[nu*k:nu*(k+1)]
                
            tk = t0 + h*k
            tk1 = t0 + h*(k + 1)
            J += 0.5*h*f_lagr(Xk, Yk, Uk, P, tk)
            J += 0.5*h*f_lagr(Xk1, Yk1, Uk, P, tk1)
        self.J = J
        # NLP codification
        w = []
        lbw = []
        ubw = []
        
        w += [tb]
        lbw += [-inf, -inf]
        ubw += [inf, inf]
        
        self.w_x0_loc = sum(arr_size(arr) for arr in w)
        w += [X]
        lbw += xlb
        ubw += xub
        
        self.w_y0_loc = sum(arr_size(arr) for arr in w)
        w += [Y]
        lbw += ylb
        ubw += yub
        
        self.w_u0_loc = sum(arr_size(arr) for arr in w)
        w += [U]
        lbw += ulb
        ubw += uub
        
        self.w_p0_loc = sum(arr_size(arr) for arr in w)
        w += [P]
        lbw += plb
        ubw += pub
        
        self.nlp_vars = w
        self.nlp_ub = ubw
        self.nlp_lb = lbw
        
        self.constraints = g
        self.constraints_ub = ubg
        self.constraints_lb = lbg
        self.problem = {'f': self.J, 'x':vertcat(*w), 'g':vertcat(*g)}
    def solve(self, ig, solver_options={}, debug_ig=False, **kwargs):
        w0 = [ig.t0, ig.tf]
        Dt = ig.tf - ig.t0
        eps = 1e-9*Dt
        t_ig = [ig.t0 + eps + (Dt - 2*eps)*k/(self.n-1) for k in range(self.n)]
        t_ig_u = [0.5*sum(t_ig[k:k+2]) for k in range(self.n - 1)]
        #t_ig_u = [ig.t0 + eps + (1-2*eps)*Dt*(0.5+k)/(self.n-1) for k in range(self.n - 1)]
        
        #t_ig_u = [ig.tu[1] + eps +(1-2*eps)*(ig.tu[-2] - ig.tu[1])*k/(self.n-2) for k in range(self.n-1)]
        for k in range(self.n):
            t = t_ig[k]
            for x_name in self.ocp.dm.x.keys():
                try:
                    x_fcn = ig.x[x_name]
                except KeyError:
                    raise KeyError(f"""The initialization trajectory provided did \
not contain an initial guess for the state '{x_name}'. \
Problem states: {self.ocp.dm.x.keys()} \
Initial guess states: {ig.x.keys()}""")
                val = x_fcn(t)
                if debug_ig:
                    print(f'{x_name}({t}) = {val}')
                w0 += [val]
        for k in range(self.n):
            t = t_ig[k]
            for y_name in self.ocp.dm.y.keys():
                w0 += [ig.y[y_name](t)]
        try:
            for k in range(self.n-1):
                t = t_ig_u[k]
                for u_name in self.ocp.dm.u.keys():
                    try:
                        u_fcn = ig.u[u_name]
                    except KeyError:
                        raise KeyError(f"""The initialization trajectory provided did \
not contain an initial guess for the control '{u_name}'. \
Problem controls: {self.ocp.dm.u.keys()} \
Initial guess controls: {ig.u.keys()}""")
                    val = u_fcn(t)
                    if debug_ig:
                        print(f'{u_name}({t}) = {val}')
                    w0 += [val]
        except ValueError:
            print("T range: ", [ig.t0, ig.tf])
            print("T sample x:", t_ig)
            print("T sample u: ", t_ig_u)
            print("T problematic: ", t)
            raise
        for p_name in self.ocp.dm.p.keys():
            try:
                w0 += [ig.p[p_name]]
            except KeyError:
                        raise KeyError(f"""The initialization trajectory provided did \
not contain an initial guess for the parameter '{p_name}'. \
Problem parameters: {self.ocp.dm.p.keys()} \
Initial guess params: {ig.p.keys()}""")
        # Solve the NLP
        self.solver = solver = nlpsol('solver', 'ipopt', self.problem, {'ipopt': solver_options})
        w0 = [float(val) for val in w0]
        sol = solver(x0=w0, lbx=self.nlp_lb, ubx=self.nlp_ub, lbg=self.constraints_lb, ubg=self.constraints_ub)
        self.sol = sol
        self.w_out = w_out = sol['x'].full().flatten()
        dt = DiscreteTrajectory()
        dt.t0 = w_out[0]
        dt.tf = w_out[1]
        dm = self.ocp.dm
        
        for j, x_name in enumerate(self.ocp.dm.x.keys()):
            dt.x[x_name] = []
        for j, name in enumerate(self.ocp.dm.y.keys()):
            dt.y[name] = []
        for j, name in enumerate(self.ocp.dm.u.keys()):
            dt.u[name] = []       
        
        for k in range(self.n):
            xk = w_out[self.w_x0_loc + k*dm.nx:self.w_x0_loc + (k + 1)*dm.nx]
            for j, x_name in enumerate(self.ocp.dm.x.keys()):
                dt.x[x_name] += [xk[j]]
        
        for k in range(self.n):
            yk = w_out[self.w_y0_loc + k*dm.ny:self.w_y0_loc + (k + 1)*dm.ny]
            for j, name in enumerate(self.ocp.dm.y.keys()):
                dt.y[name] += [yk[j]]
                
        for k in range(self.n-1):
            uk = w_out[self.w_u0_loc + k*dm.nu:self.w_u0_loc + (k + 1)*dm.nu]
            for j, name in enumerate(self.ocp.dm.u.keys()):
                dt.u[name] += [uk[j]]
                
        p = w_out[self.w_p0_loc:self.w_p0_loc + dm.np]
        for j, name in enumerate(self.ocp.dm.p.keys()):
            dt.p[name] = p[j]
            
        dt.t = [dt.t0 + (dt.tf - dt.t0)*k/(self.n-1) for k in range(self.n)]
        dt.tu = [dt.t0 + (dt.tf - dt.t0)*(k + 0.5)/(self.n-1) for k in range(self.n-1)]
        
        dt.status = solver.stats()['return_status']
        dt.solver_stats = solver.stats()
        dt.J = float(sol['f'])
        return dt




