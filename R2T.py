import psycopg2
import numpy as np
from pulp import *
import multiprocessing as mp
from functools import partial

EPSILON = 0.8
BETA = 0.1
GSQ = 16384

multi_private_query = """
SELECT supplier.s_suppkey, customer.c_custkey, l_extendedprice * (1 - l_discount) / 100
FROM supplier, lineitem, orders, customer
WHERE supplier.s_suppkey = lineitem.l_suppkey AND lineitem.l_orderkey = orders.o_orderkey
AND orders.o_custkey = customer.c_custkey AND orders.o_orderdate >= '1997-07-01'
"""

single_private_query = """
SELECT customer.c_custkey, l_extendedprice * (1 - l_discount) / 100
FROM supplier, lineitem, orders, customer
WHERE supplier.s_suppkey = lineitem.l_suppkey AND lineitem.l_orderkey = orders.o_orderkey
AND orders.o_custkey = customer.c_custkey AND orders.o_orderdate >= '1997-07-01'
"""

def get_db_connection():
    return psycopg2.connect(
        dbname="tpch1g",
        user="postgres",
        password="Yujinqi.2002",
        host="localhost",
        port="5432"
    )

def laplace(sensitivity, epsilon) -> float:
    noise = np.random.laplace(0, sensitivity / epsilon)
    return noise

def get_threshold(gs_Q):
    n = (np.log2(gs_Q))+1
    return gs_Q / (2**np.arange(n))

class R2T():
    def __init__(self, epsilon=EPSILON, gs_Q=GSQ):
        self.epsilon = epsilon
        self.gs_Q = gs_Q
         
    def aux_query(self, conn, query):
        cur = conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        return results
        
    def get_C_j(self, results):
        C = {}
        for i, row in enumerate(results):
            key = tuple(row[:-1])
            if key not in C:
                C[key] = []
            C[key].append(i)
        return C
        
    def get_D_l(self, proj_results, results):
        D = {}
        # For each projected result (pl)
        for l, proj_val in enumerate(proj_results):
            D[l] = []
            # Find all indices j where pl = πy qj(I)
            for j, result in enumerate(results):
                if proj_val[0:-1] == result[0:-1]:
                    D[l].append(j)
        return D

    def solve_LP(self, results, threshold):
        prob = LpProblem("Maximize_Q", LpMaximize)
        u = LpVariable.dicts("u", 
                            ((k) for k in range(len(results))), 
                            lowBound=0,
                            upBound=None)
        
        C = self.get_C_j(results)
        
        prob += lpSum([u[k] for k in range(len(results))])
        
        for j, KrefJ in C.items():
            prob += lpSum([u[k] for k in KrefJ]) <= threshold
            
        for k in range(len(results)):
            prob += u[k] <= float(results[k][-1])
        
        prob.solve(PULP_CBC_CMD(msg=0))
        return value(prob.objective)
    
    def solve_dual_LP(self, results, threshold):
        prob = LpProblem("Minimize_Dual", LpMinimize)

        C = self.get_C_j(results)  # C is a dict {j: [k1, k2, ...], ...}

        y = LpVariable.dicts("y", C.keys(), lowBound=0)

        z = LpVariable.dicts("z", (k for k in range(len(results))), lowBound=0)

        b = [float(results[k][-1]) for k in range(len(results))]

        prob += (
            lpSum([y[j] * threshold for j in C.keys()]) +
            lpSum([z[k] * b[k] for k in range(len(results))])
        ), "DualObjective"

        for k in range(len(results)):
            # Identify all j such that k is in KrefJ_j
            relevant_js = [j for j, KrefJ_j in C.items() if k in KrefJ_j]
            # Define the constraint: sum(y_j for relevant j) + z_k >= 1
            prob += (
                lpSum([y[j] for j in relevant_js]) + z[k] >= 1
            ), f"DualConstraint_{k}"

        prob.solve(PULP_CBC_CMD(msg=0))
        return value(prob.objective)
    
    def solve_proj_LP(self, results, proj_results, threshold):
        prob = LpProblem("Minimize_Proj", LpMaximize)

        C = self.get_C_j(results)  # C is a dict {j: [k1, k2, ...], ...}
        D = self.get_D_l(proj_results, results)  # D is a dict {l: [j1, j2, ...], ...}
        
        v = LpVariable.dicts("v", 
                            ((l) for l in range(len(proj_results))), 
                            lowBound=0,
                            upBound=None)
        
        u = LpVariable.dicts("u", 
                            ((k) for k in range(len(results))), 
                            lowBound=0,
                            upBound=None)
        
        prob += lpSum([v[l] for l in range(len(proj_results))])
        
        for j, kRefL in D.items():
            prob += v[j] <= lpSum([u[k] for k in kRefL])

        for j, KrefJ in C.items():
            prob += lpSum([u[k] for k in KrefJ]) <= threshold
        
        for k in range(len(results)):
            prob += u[k] <= float(results[k][-1])
            
        for l in range(len(proj_results)):
            prob += v[l] <= float(proj_results[l][-1])
         
        prob.solve(PULP_CBC_CMD(msg=0))
        return value(prob.objective)
        
    # def solve_LP_CPLEX(self, results, threshold):
    #     prob = cplex.Cplex()

    #     prob.objective.set_sense(prob.objective.sense.maximize)

    #     prob.set_log_stream(None)
    #     prob.set_error_stream(None)
    #     prob.set_warning_stream(None)
    #     prob.set_results_stream(None)

    #     # create variables
    #     var_names = [f'u_{k}' for k in range(len(results))]
    #     obj_coeffs = [1.0] * len(results)  
    #     lb = [0.0] * len(results)         
    #     ub = [float(results[k][-1]) for k in range(len(results))]  

    #     prob.variables.add(obj=obj_coeffs,
    #                       lb=lb,
    #                       ub=ub,
    #                       names=var_names)

    #     C = self.get_C_j(results)

    #     constraint_count = 0
    #     for j, KrefJ in C.items():
    #         if KrefJ:
    #             constraint_vars = [f'u_{k}' for k in KrefJ]
    #             constraint_coeffs = [1.0] * len(KrefJ)

    #             prob.linear_constraints.add(
    #                 lin_expr=[[constraint_vars, constraint_coeffs]],
    #                 senses=['L'],  
    #                 rhs=[threshold],  
    #                 names=[f'c_{constraint_count}']
    #             )
    #             constraint_count += 1

    #     prob.solve()

    #     return prob.solution.get_objective_value()
    
    # def solve_dual_LP_CPLEX(self, results, threshold):
    #     prob = cplex.Cplex()

    #     prob.objective.set_sense(prob.objective.sense.minimize)

    #     prob.set_log_stream(None)
    #     prob.set_error_stream(None)
    #     prob.set_warning_stream(None)
    #     prob.set_results_stream(None)

    #     C = self.get_C_j(results)

    #     var_names = []
    #     obj_coeffs = []

    #     for j in C.keys():
    #         var_names.append(f'y_{j}')
    #         obj_coeffs.append(threshold)

    #     for k in range(len(results)):
    #         var_names.append(f'z_{k}')
    #         obj_coeffs.append(float(results[k][-1]))

    #     prob.variables.add(obj=obj_coeffs,
    #                       names=var_names,
    #                       lb=[0.0] * len(var_names))

    #     for k in range(len(results)):
    #         constraint_coeffs = []
    #         constraint_vars = []

    #         for j, KrefJ in C.items():
    #             if k in KrefJ: 
    #                 constraint_vars.append(f'y_{j}')
    #                 constraint_coeffs.append(1.0)

    #         constraint_vars.append(f'z_{k}')
    #         constraint_coeffs.append(1.0)

    #         prob.linear_constraints.add(
    #             lin_expr=[[constraint_vars, constraint_coeffs]],
    #             senses=['E'],
    #             rhs=[1.0],
    #             names=[f'c_{k}']
    #         )

    #     prob.solve()

    #     return prob.solution.get_objective_value()

    def Q_under_threshold(self, results, proj_results, threshold):
        T = laplace(np.log2(self.gs_Q)*threshold, self.epsilon) - \
            np.log2(self.gs_Q) * np.log(np.log2(self.gs_Q)/BETA)*threshold/self.epsilon
        
        if proj_results is not None:
            Q_threshold = self.solve_proj_LP(results, proj_results, threshold)
        else:
            Q_threshold = self.solve_dual_LP(results, threshold)

        return Q_threshold + T, Q_threshold
    
    def R2T_truncate(self, results, proj_results=None):
        Q_hat = 0
        
        # For Debugging
        # Q_hat, Q_threshold = self.Q_under_threshold(results, proj_results, self.gs_Q)        
        # print("Q_threshold:" + str(Q_threshold))
        
        thresholds = get_threshold(self.gs_Q)

        pool = mp.Pool(processes=min(8, len(thresholds)))
        
        process_func = partial(self.Q_under_threshold, results, proj_results)
        Qresults = pool.map(process_func, thresholds)
        
        pool.close()
        pool.join()
        
        print(Qresults)
        Q_hat, Q_threshold = max(Qresults, key=lambda x: x[0])
        
        return Q_hat, Q_threshold

    def R2T_truncate_early_stop(self, results, proj_results=None):
        Q_hat = 0
        
    def query(self, conn, aux_query, aux_proj_query=None):
        aux_results = self.aux_query(conn, aux_query)
        
        aux_proj_results = None
        if aux_proj_query is not None:
            aux_proj_results = self.aux_query(conn, aux_proj_query)
            
        Q_hat, _ = self.R2T_truncate(aux_results, aux_proj_results)

        return Q_hat

if __name__ == "__main__":
    conn = get_db_connection()
    r2t = R2T()
    result = r2t.query(conn, multi_private_query)
    conn.close()
    print("R2T Q:" + str(result))