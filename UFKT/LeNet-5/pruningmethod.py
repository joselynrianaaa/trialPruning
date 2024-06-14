
import torch as th
import torch.nn as nn 
import numpy as np
from itertools import chain
from collections import defaultdict


class PruningMethod():
   
    def prune_filters(self,indices):
      conv_layer=0
      print(self)

      for layer_name, layer_module in self.named_modules():
        if(isinstance(layer_module, th.nn.Conv2d)):
          if(conv_layer==0):            
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            
          else:
            in_channels=indices[conv_layer-1]

          out_channels=indices[conv_layer]
          layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))
          #layer_module.weight = layer_module.weight.data.cpu().numpy()[out_channels]

          if layer_module.bias is not None:
            #layer_module.bias   = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias   = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))


          layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:,in_channels])).to('cuda'))

          '''print(state[optimizer_count]['momentum_buffer'].shape,'/////////')
          optimizer_momentum=state[optimizer_count]['momentum_buffer'].data.cpu().numpy()
          optimizer_momentum=th.FloatTensor(th.from_numpy(optimizer_momentum[out_channels]))
          state[optimizer_count]['momentum_buffer']=optimizer_momentum[:,in_channels]
          print('conv',state[optimizer_count]['momentum_buffer'].shape)
          optimizer_count+=1
          optimizer_momentum=state[optimizer_count]['momentum_buffer'].data.cpu().numpy()
          state[optimizer_count]['momentum_buffer']=th.FloatTensor(th.from_numpy(optimizer_momentum[out_channels]))
          print('bias',state[optimizer_count]['momentum_buffer'].shape)
          optimizer_count+=1'''
          
          layer_module.in_channels=len(in_channels)
          layer_module.out_channels=len(out_channels)
          conv_layer+=1
          #print(self)
        
        if (isinstance(layer_module, th.nn.BatchNorm2d)):
            out_channels=indices[conv_layer]

            layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

            layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
            layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
            
            '''optimizer_momentum=state[optimizer_count]['momentum_buffer'].data.cpu().numpy()
            state[optimizer_count]['momentum_buffer']=th.FloatTensor(th.from_numpy(optimizer_momentum[out_channels]))
            print('BN',state[optimizer_count]['momentum_buffer'].shape)
            optimizer_count+=1
            optimizer_momentum=state[optimizer_count]['momentum_buffer'].data.cpu().numpy()
            state[optimizer_count]['momentum_buffer']=th.FloatTensor(th.from_numpy(optimizer_momentum[out_channels]))
            print('BN_bias',state[optimizer_count]['momentum_buffer'].shape)
            optimizer_count+=1'''

            layer_module.num_features= len(out_channels)

        if isinstance(layer_module, nn.Linear):
            conv_layer-=1
            in_channels=indices[conv_layer]

            weight_linear = layer_module.weight.data.cpu().numpy()

            size=4*4
            expanded_in_channels=[]
            for i in in_channels:
              for j in range(size):
                expanded_in_channels.extend([i*size+j])

            layer_module.weight = th.nn.Parameter(th.from_numpy(weight_linear[:,expanded_in_channels]).to('cuda'))

            '''print(state[optimizer_count]['momentum_buffer'].shape,'/////////')
            optimizer_momentum=state[optimizer_count]['momentum_buffer'].data.cpu().numpy()
            #optimizer_momentum=th.FloatTensor(th.from_numpy(optimizer_momentum[out_channels]))
            state[optimizer_count]['momentum_buffer']=optimizer_momentum[:,expanded_in_channels]
            print('FC',state[optimizer_count]['momentum_buffer'].shape)
            optimizer_count+=1'''

            layer_module.in_features = len(expanded_in_channels)
            break

           

      
    def get_indices_topk(self,layer_bounds,i,prune_limit, prune_percentage):

      indices=int(len(layer_bounds)*prune_percentage[i])+1 
      p=len(layer_bounds) 
      if (p-indices)<prune_limit: 
         remaining=p-prune_limit
         indices=remaining
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      return k

    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k



          





  

