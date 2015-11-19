require 'torch'
require 'nn'
require 'gnuplot'
require 'nngraph'
require 'optim'
require 'lfs'
require 'pca2.lua'
require 'time'
m = require 'manifold'
--require 'plot_factory'
LSTM = require 'model.lua'
local model_utils=require 'model_utils'

local params = {batch_size=20,
                seq_length=5,
                layers=2,
                size_of_input=530, 
                size_of_output=527, --Size_of_output = size_of_input - number_of_network_atributes_expect_tfidf 
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                number_of_inputs=4345, --4345
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
                save_every=1111,
                print_every=1
              }



-- Initalization of network
local size_of_networks = {250,250,527,527}

-- 3-layer LSTM network (input and output have 3 dimensions)
network = {LSTM.create(params.size_of_input, 250), LSTM.create(250, 250), LSTM.create(250, params.size_of_output),nn.LogSoftMax()}

local para, grad_params = model_utils.combine_all_parameters(network[1], network[2], network[3])
para:uniform(-0.08, 0.08)

-- Network input previous states
previous_state = {
  {torch.zeros(1, 250), torch.zeros(1,250)},
  {torch.zeros(1, 250), torch.zeros(1,250)},
  {torch.zeros(1, 527), torch.zeros(1,527)}
}

local criterion = nn.ClassNLLCriterion()

-- Load Data from CSV
function load()
  -- Read data from CSV to tensor
  local csvFile = io.open("test.csv", 'r')  

  data = torch.Tensor(params.number_of_inputs, params.size_of_input)
  local id = ""
  local i = 0  
  number_of_valid_inputs = 0

  for line in csvFile:lines('*l') do  
   
    local row = line:split('\t')

    -- Update number of access
    if id ~= row[4] then
      id = row[4]
      i = 0
    end

    local l = row[10]:split(';')
    if row[10] ~= "Nil" then
       i = i + 1
       number_of_valid_inputs = number_of_valid_inputs + 1

      for key, val in ipairs(l) do
        if key == 1 then
          data[number_of_valid_inputs][key] = i -- Number of access
          data[number_of_valid_inputs][key+1],data[number_of_valid_inputs][key+2] = please_get_sin_and_cos(row[1]) -- Time in the sinus format and the cosinus fromat
        else
          data[number_of_valid_inputs][key+2] = tonumber(val) -- Tfidf value
        end  
        
      end
    end
  end
end

function save()
  local file = io.open("result.tsv", 'w') 

  for i=1, number_of_valid_inputs do
    str_data = "V,"
    str_pred = "P,"
      
   for y=1, params.size_of_output do
      str_data = str_data..string.format("%s",data[i][y+3]..";")
      str_pred = str_pred..string.format("%s",predicted_data[i][y]..';')  
   end
   str_data = str_data.."\n"
   str_pred = str_pred.."\n"

    file:write(str_data)
    file:write(str_pred)
  end

end

load()
selector = 1
predicted_data = torch.Tensor(params.number_of_inputs, params.size_of_output)
print(number_of_valid_inputs)

-- Forward and Backward pass through network
function fbbf(para_)
  if para_ ~= para then
      para:copy(para_)
  end
    grad_params:zero()
 
  local output = nil
  local input = {}
  local loss = 0
  

  -- Iteration 
  for y = selector, params.seq_length+selector-1 do
    
    -- Network input
    data_input = data[y]:resize(1,params.size_of_input)

    -- Forward pass through networks layers 
    layer_input = {data_input, table.unpack(previous_state[1])}
    for l = 1, #network-1 do
      -- Forward the input
      layer_output = network[l]:forward(layer_input)
      
      -- Save state for next iteration
      for n = 1, #previous_state[l] do
        previous_state[l][n] = layer_output[n]
      end    
      
      -- print(previous_state[l][1])
      -- print(layer_output[1])
      table.insert(input, layer_input)
      
      -- extract hidden state from output
     -- print(string.format("output is %s",layer_output))

      --print(layer_output[2][1])
      --print(data[y+1])
      local layer_h = layer_output[2]
      -- prepare next layer's input or set the output
      if l < #network-1 then
        layer_input = {layer_h, table.unpack(previous_state[l + 1])}
      else
        output = layer_h
      end
    end
    
    output = network[4]:forward(output)

  --  local target = data[y+1]:resize(1,params.size_of_input)
    --print(data[y+1])
   -- Reduciton dimension. It takes only tfidf.
    local target = data[y+1]:narrow(1,4,params.size_of_output)
    target = target:resize(1,params.size_of_output)
    --print(target)
    --print(output:size())
    --print(target:size())
    target = target:select(1,1)
    output = output:select(1,1)
      
    --output = torch.Tensor({output})

    --target = torch.Tensor({target}) 

    print(output)
    --print(target)
    loss = loss + criterion:forward(output, target)
    --print(predicted_data[y])
    print(predicted_data[y]:size())
    predicted_data[y] = output
    
    dOutput = criterion:backward(output, target) 
    
  end

  --visual_data(data,predicted_data,params)

  -- backward pass through networks layers 
  for t = #network, 1, -1 do -- reversed loop
      local derr = torch.ones(1,size_of_networks[t])
      local dInputT = {derr,dOutput[1]}
      --print(string.format("GrandOutpu is %s",dInputT)) 
      --print(string.format("Network is %s",#network[t].outnode.children))
     -- print(#dInputT ~= #network[t].children.nodes)
     
      --dOutputT  = torch.Tensor(dOutputT)
      dOutput = network[t]:backward(input[t], dInputT) 
      network[t]:updateParameters(1e-1) 
  end

  -- clip gradient element-wise
  grad_params:clamp(-5, 5)
 -- print(loss)
  selector = selector + params.seq_length
  return loss, grad_params
end

-- optimization stuff

local losses = {}
--nngraph.setDebug(true)
local optim_state = {learningRate = 1e-1}
local iterations = 868
for i = 1, iterations do
    local _, loss = optim.adagrad(fbbf, para, optim_state)
    losses[#losses + 1] = loss[1]

    if i % params.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % params.print_every == 0 then
        --print(loss[i])
        --print(params.seq_length)
       print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / params.seq_length, grad_params:norm()))
    end
end
predicted_data = predicted_data:sub(4330,4340,1,527)
print(predicted_data)
p = m.embedding.tsne(predicted_data, {dim=2, perplexity=30})  
gnuplot.plot{
     {'original',p:squeeze(),'+'}
    }
    gnuplot.axis('equal')
    gnuplot.axis{-20,50,-10,30}
    gnuplot.grid(true)
    local answer
    repeat
       io.write("continue with this operation (y/n)? ")
       io.flush()
       answer=io.read()
    until answer=="y" or answer=="q"

--save()
