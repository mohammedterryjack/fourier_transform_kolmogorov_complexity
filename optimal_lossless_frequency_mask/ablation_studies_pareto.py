from pareto_front import plot_pareto_front
from compare_pareto_front import plot_pareto_front_two_rules

if __name__ == "__main__":
   rules = (11, 110)
   params = dict(
       ic=111,
       width=20,
       height=20,
       sigmoid_sharpness=1,
       learning_rate=1e-2,
       sparsity_loss_weight=1,
       iterations=500,
       quantisation_threshold=0.5
   )
   ablations = dict(
     quantisation_threshold=[0.1, 0.3, 0.5, 0.7, 0.9],
     iterations=[10,50,100,500,1000,5000],
     sparsity_loss_weight=[0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0],
     learning_rate=[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3],
     sigmoid_sharpness=[0.001,0.01,0.1,1,10,100],
     width=[10,50,100,500,1000],
   ) 
   for key,values in ablations.items():
      params['key']=key
      params['values']=values   

      for rule in rules: 
          plot_pareto_front(
             rule=rule,
             **params 
          )
      plot_pareto_front_two_rules(
          rules=rules,
          **params 
      )
