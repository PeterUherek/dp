require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'pca2.lua'
require 'gnuplot'

-- utils
--require 'utils.functions'
--require 'utils.lfs'
local params = {batch_size=20,
                seq_length=5,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
                save_every=1111,
                print_every=1
              }

function load()
  -- Read data from CSV to tensor
  local csvFile = io.open("test.csv", 'r')  
  local header = csvFile:read()

  data = torch.Tensor(params.seq_length, 528)

  local i = 0  
  for line in csvFile:lines('*l') do  
    i = i + 1
    local l = line:split(';')
    for key, val in ipairs(l) do
       if key == 1 then
        data[i][key] = i
        else
        data[i][key] = val
        end
    end
  end

  csvFile:close()
  return data  
end


local size_of_networks = {250,250,528}

LSTM = require 'dp.lua'
local model_utils=require 'model_utils'

-- 3-layer LSTM network (input and output have 3 dimensions)
network = {LSTM.create(528, 250), LSTM.create(250, 250), LSTM.create(250, 528)}

local para, grad_params = model_utils.combine_all_parameters(network[1], network[2], network[3])
para:uniform(-0.08, 0.08)

-- network input previous states
previous_state = {
  {torch.zeros(1, 250), torch.zeros(1,250)},
  {torch.zeros(1, 250), torch.zeros(1,250)},
  {torch.zeros(1, 528), torch.zeros(1,528)}
}

function fbbf(para_)
  if para_ ~= para then
      para:copy(para_)
  end
    grad_params:zero()

  local output = nil
  local next_state = torch.Tensor(params.seq_length, 528)
  local input = {}
  local loss = 0
  local data = load()
 -- print(data)
  -- iteration
  for y = 1, params.seq_length-1 do
      print(y)
    -- network input
    --local x = torch.randn(1, 528)
    local criterion = nn.MSECriterion()
     --print(x)
    data_input = data[y]:resize(1,528)
    -- forward pass through networks layers 
    layer_input = {data_input, table.unpack(previous_state[1])}
    for l = 1, #network do
      -- forward the input
      layer_output = network[l]:forward(layer_input)
      
      -- save state for next iteration
      for n = 1, #previous_state[l] do
        previous_state[l][n] = layer_output[n]
      end    
      
      -- print(previous_state[l][1])
      -- print(layer_output[1])
      table.insert(input, layer_input)
      
      -- extract hidden state from output
      print(string.format("output is %s",layer_output))

      --print(layer_output[2][1])
      --print(data[y+1])
      local layer_h = layer_output[2]
      -- prepare next layer's input or set the output
      if l < #network then
        layer_input = {layer_h, table.unpack(previous_state[l + 1])}
      else
        output = layer_h
      end
    end

    local target = data[y+1]:resize(1,528)
   -- local target = data[y+1]:resize(1,528)
    loss = loss + criterion:forward(output, target)

    next_state[y]=output
    dOutput = criterion:backward(layer_output[2], target) 
  end

print(next_state)
  --print(data)
  data = data:resize(528,5)
 --next_state = torch.Tensor(next_state)
  print(next_state)
  next_state = next_state:resize(528,5)

  original_reduced = pca(data,3,100)
  predicted = pca(next_state,3,100)

    gnuplot.plot{
     {'original',original_reduced:squeeze(),'+'},
     {'predicted',predicted:squeeze(),'+'}
    }
    gnuplot.axis('equal')
    gnuplot.axis{-20,50,-10,30}
    gnuplot.grid(true)
    print(original_reduced)
    local answer
    repeat
       io.write("continue with this operation (y/n)? ")
       io.flush()
       answer=io.read()
    until answer=="y" or answer=="q"

  -- local dRnnState = {[#network] = utils.cloneList(previous_state, true)}
  -- print(dRnnState)

  --local input = torch.Tensor(1,3):fill(1)
 

  --print(string.format("target is %s",target))
   --print(next_state)
   --print(input)
 
  -- backprop through loss, and softmax/linear

    -- backward pass through networks layers 
  for t = #network, 1, -1 do -- reversed loop
      local derr = torch.ones(1,size_of_networks[t])
      local dInputT = {derr,dOutput[1]}
      --print(string.format("GrandOutpu is %s",dInputT)) 
      --print(string.format("Network is %s",#network[t].outnode.children))
     -- print(#dInputT ~= #network[t].children.nodes)
     
      --dOutputT  = torch.Tensor(dOutputT)
      dOutput = network[t]:backward(input[t], dInputT)  

      print("Tuto bude loss")
     -- print(loss)
      
      print(dInputT)
  end

  -- clip gradient element-wise
  grad_params:clamp(-5, 5)
  print(loss)
  return loss, grad_params
end

-- optimization stuff
local losses = {}
--nngraph.setDebug(true)
local optim_state = {learningRate = 1e-1}
local iterations = 1
for i = 1, iterations do
    local _, loss = optim.adagrad(fbbf, para, optim_state)
    losses[#losses + 1] = loss[1]

    if i % params.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % params.print_every == 0 then
        print(loss[i])
        print(params.seq_length)
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / params.seq_length, grad_params:norm()))
    end
end