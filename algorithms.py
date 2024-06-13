"""Different algorithms related to gradient descent used to optimize the actions on different kind of games. Only the first one, IPGA_stage_game
   is used in run.ipynb 
   AUTHORS: Gabriel Vallat
"""
import torch as th
import numpy as np

def IPGA_stage_game(game,init_pol, iter=10000,lr=None):
    """Run Independent policy gradient ascent on the given game with the given initial policy or policies

    Args:
        game (game_definition.StageGame): The game to solve
        init_pol (torch.tensor): The initial policy or policies. If batched, the tensor should have a shape of (batch_size, n_players, n_action) 
        iter (int, optional): maximum number of iteration to run. Defaults to 10000.
        lr (float, optional): learning rate. Defaults to None. If None, the learning rate is chosen as 1/(game.na*(game.n-1))

    Returns:
        _type_: _description_
    """
    pol = th.clone(init_pol)
    flag=False
    pol.requires_grad=True
    lr = lr or 1/(game.na*(game.n-1))
    opt = th.optim.SGD([pol], lr=lr,maximize=True)
    
    best_pol = th.tensor(np.unravel_index(th.argmax(game.r).cpu(),game.r.shape),device=game.device)
    batched = pol.dim()!=2
    r_prev = 0
    #print(f"Result:{th.abs(pol-th.nn.functional.one_hot(best_pol,num_classes=game.na)).mean()}",end="\r")
    for i in range(iter):
        opt.zero_grad()
        if batched:
            r = game.solve_stoc_pol_batch(pol).sum()
        else:
            r = game.solve_stoc_pol(pol)
        if r_prev > r:
            print("Warning: learning rate to large with respect to the smoothness")
        r.backward()
        opt.step()
        #project back into the simplex
        pol.data=project_simplex_batch(pol)
        #print(f"Result:{th.abs(pol-th.nn.functional.one_hot(best_pol,num_classes=game.na)).mean()}",end="\r")
        #print(f"{r=}",end="\r")
        m,argm = pol.max(-1)
        #early stops if the policy is in the corner of the simplex for two steps in a row
        if (m ==1).all():
            if flag:
                break
            flag=True
        else:
            flag=False
            
    return pol

def IPGA_dynamic_game_oracle(game,init_pol,s_0=None, iter=1000):
    pol = th.clone(init_pol)
    pol.requires_grad=True
    opt = th.optim.SGD([pol], lr=0.05,maximize=True)
    #print(f"Result:{th.abs(pol-th.nn.functional.one_hot(best_pol,num_classes=game.na)).mean()}",end="\r")
    for i in range(iter):
        opt.zero_grad()
        V = game.play_game(pol,s_0)
        V.mean().backward()
        opt.step()
        #project back into the simplex
        pol.data = project_simplex_batch(pol)
        print(f"{V=}",end="\r")

        m,argm = pol.max(-1)
        if (m ==1).all():
            break
    return pol,V
def PGA_dynamic_game_oracle(game,init_pol,s_0=0, iter=100000):
    pol = th.clone(init_pol)
    pol_central = th.ones([game.ns]+[game.na]*game.n,device=game.device)
    for p in range(game.n):
        shape = th.ones(game.n+1,dtype=int)
        shape[0] = game.ns
        shape[p+1]=game.na
        pol_central*=pol[p].reshape(tuple(shape))

    pol_central = pol_central.flatten(1)
    pol_central.requires_grad=True
    opt = th.optim.SGD([pol_central], lr=0.1,maximize=True)
    #print(f"Result:{th.abs(pol-th.nn.functional.one_hot(best_pol,num_classes=game.na)).mean()}",end="\r")
    for i in range(iter):
        opt.zero_grad()
        V = game.play_game_CCE(pol_central,s_0=s_0)
        V.mean().backward()
        opt.step()
        #project back into the simplex
        pol_central.data = project_simplex_batch(pol_central)
        print(f"{V=}",end="\r")
        m,argm = pol_central.max(-1)
        if (m ==1).all():
            break
    end_pol = np.array(np.unravel_index(argm.cpu().numpy(),[game.na]*game.n)).T
    pol = th.nn.functional.one_hot(th.tensor(end_pol,device=game.device).T,num_classes=game.na)
    return pol,V
def SIPGA_dynamic_game_play(game,init_pol,s_0=0, iter=100000,stable_target=False,optimistic_V=False,share_pol=False):
    """Can be seen as a directly parametrized actor-critic"""
    pol = th.clone(init_pol)
    if optimistic_V:
        V = 1/(1-game.gamma)*th.ones(game.ns,device=game.device)
    else:
        V = th.zeros(game.ns,device=game.device)
    V.requires_grad=True
    #the depedence on the policy is linear, so no optimizer is required
    #pol.requires_grad=True
    lr_pol = 0.01
    #opt_pol = th.optim.SGD([pol], lr=0.01,maximize=True)
    opt_V = th.optim.SGD([V], lr=0.01,maximize=False)
    ns = s_0
    for i in range(iter):
        opt_V.zero_grad()

        s = ns
        actions = [game.random_np.choice(np.arange(game.na),p=pol[p,s].detach().cpu().numpy()) for p in range(game.n)]
        
        ns,r = game.play_step(s,actions)
        if stable_target:
            V_s=  r+game.gamma*V[ns].detach()#detach to not go twice through the graph
        else:
            V_s=  r+game.gamma*V[ns]

        action_ar = np.array(actions)

        if share_pol:
            #should not be used: the pol is already embedded in V
            allp = th.arange(game.n)
            for p in range(game.n):
                notp = th.cat([allp[:0],allp[1:]]) 
                polnotp = pol[notp,s*th.ones(game.n-1,dtype=int),action_ar[notp]]
                pol[p,s,action_ar[p]] += lr_pol*V_s*polnotp.prod()
        else:
            pol[th.arange(game.n),s*th.ones(game.n,dtype=int),action_ar] += lr_pol*V_s
        err = th.square(V_s-V[s])

        err.backward()
        opt_V.step()
        
        #project back into the simplex
        pol.data = project_simplex_batch(pol)
        print(f"{V=}",end="\r")

        m,argm = pol.max(2)
        if (m ==1).all():
            break
    return pol,V
def SPGA_dynamic_game_play(game,init_pol,s_0=0, iter=100000,stable_target=True,optimistic_V=False):
    """Find a centralized solution (a coarse correlated equilibrium)"""
    pol = th.clone(init_pol)
    pol_central = th.ones([game.ns]+[game.na]*game.n,device=game.device)
    for p in range(game.n):
        shape = th.ones(game.n+1,dtype=int)
        shape[0] = game.ns
        shape[p+1]=game.na
        pol_central*=pol[p].reshape(tuple(shape))

    pol_central = pol_central.flatten(1)
    if optimistic_V:
        V = 1/(1-game.gamma)*th.ones(game.ns,device=game.device)
    else:
        V = th.zeros(game.ns,device=game.device)
    V.requires_grad=True
    #the depedence on the policy is linear, so no optimizer is required
    #pol.requires_grad=True
    lr_pol = 0.01
    #opt_pol = th.optim.SGD([pol], lr=0.01,maximize=True)
    opt_V = th.optim.SGD([V], lr=0.01,maximize=False)
    ns = s_0
    for i in range(iter):
        opt_V.zero_grad()

        s = ns
        actions_id = game.random_np.choice(np.arange(pol_central[s].numel()),p=pol_central[s].cpu().numpy())
        actions = np.unravel_index(actions_id,[game.na]*game.n)
        ns,r = game.play_step(s,actions)
        if stable_target:
            V_s=  r+game.gamma*V[ns].detach()#detach to not go twice through the graph
        else:
            V_s=  r+game.gamma*V[ns]

        action_ar = np.array(actions)

        
        pol_central[s,actions_id]  += lr_pol*V_s

        err = th.square(V_s-V[s])

        err.backward()
        opt_V.step()
        
        #project back into the simplex
        pol_central.data[s] = project_simplex_batch(pol_central[s])
        print(f"{V=}",end="\r")
        m,argm = pol_central.max(1)
        if (m ==1).all():
            break
    #note that we go to a greedy policy at the end, as we cannot necessarly convert the probability over the joint policies to the decentralized ones
    end_pol = np.array(np.unravel_index(argm.cpu().numpy(),[game.na]*game.n)).T
    pol = th.nn.functional.one_hot(th.tensor(end_pol,device=game.device).T,num_classes=game.na)
    return pol,V
def project_simplex(x,tol=1e-6,max_steps=1000000):
    mu = 0
    pmu = 1
    #use Newton method on h to find mu
    s=0
    while abs(mu-pmu)>tol and s<max_steps:
        s+=1
        pmu=mu
        mu = mu + (th.clip(x-mu,0).sum()-1)/((x-mu)>0).sum()
    proj_x = th.clip(x-mu,0) 
    if s==max_steps:
        print("Warning: the projection did not converge")
    return proj_x
def project_simplex_batch(x,tol=1e-6,max_steps=100):
    mu = th.zeros(x.shape[:-1],device=x.device).unsqueeze(-1)
    pmu = th.ones(x.shape[:-1],device=x.device).unsqueeze(-1)
    #use Newton method on h to find mu
    s=0
    
    while (mu-pmu).abs().max()>tol and s<max_steps:
        s+=1
        pmu=mu
        mu = mu + (th.clip(x-mu,0).sum(-1,keepdim=True)-1)/((x-mu)>0).sum(-1,keepdim=True)
    if s==max_steps:
        print("Warning: the projection did not converge")
    proj_x = th.clip(x-mu,0) 
    return proj_x
def test_stage():
    import game_definition
    game = game_definition.StageGame(3,10,coupling_w=0.5,seed=231242351,device="cuda")
    init_pol = th.rand(100,game.n,game.na,device=game.device,generator=game.gen)
    init_pol=init_pol/init_pol.sum(-1)[...,None]
    end_pol = IPGA_stage_game(game,init_pol=init_pol)
    greedy_pol_oh, greedy_r = game.solve_greedy_pol_batch(end_pol)
    polnp = np.argmax(end_pol.detach().cpu().numpy(),1)
    print("test finished")
    #print(f"converged to a nash: {game.is_nash(polnp)}                     ")
def test_dynamic():
    import game_definition
    game = game_definition.MultiStageGame(10,2,2,
                                          mult_r_w=0,
                                          controlable_t_w=1,
                                          seed=1,device="cuda")
    #init_pol = th.rand(game.n,game.ns, game.na,device=game.device,generator=game.gen)
    init_pol = th.ones(game.n,game.ns, game.na,device=game.device)
    init_pol=init_pol/init_pol.sum(-1)[...,None]
    s_0 =1
    end_pol,V = IPGA_dynamic_game_oracle(game,init_pol=init_pol,s_0=s_0)
    print(f"Policy gradient V estimation: {V}                        ")
    V_end_pol = game.play_game(end_pol)
    print(f"Policy gradient V real:       {V_end_pol[s_0]}                        ")
    m,argm = end_pol.max(-1)
    greedy_pol = th.nn.functional.one_hot(argm,num_classes=game.na)
    V_end_pol_greedy = game.play_game(greedy_pol)
    print(f"Greedy Policy gradient V real:{V_end_pol_greedy[s_0]}                        ")
    #polnp = np.argmax(end_pol.detach().cpu().numpy(),1)
    opt_pol, opt_v = game.solve()
    print(f"Policy iteration V:           {opt_v[s_0]}")
    print(f"Number of wrong actions: {(opt_pol!=greedy_pol).sum()/2}")
    pass
if __name__ =="__main__":
    test_stage()