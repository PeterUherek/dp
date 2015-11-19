require 'torch'
require 'nn'
require 'gnuplot'
require 'nngraph'
require 'lfs'
require 'time'
require 'RNN'
m = require 'manifold'
--require 'plot_factory'

local threshold = 4310  -- train up to this offset in the file, then predict.
local inputSize = 530       -- the number features in the csv file to use as inputs 
local hiddenSize = 529     -- the number of hidden nodes
local hiddenSize2 = 527     -- the second hidden layer
local outputSize = 527      -- the number of outputs representing the prediction targat. 
local dropoutProb = 0.6   -- a dropout probability, might help with generalisation
local rho = 2000            -- the timestep history we'll recurse back in time when we apply gradient learning
local batchSize = 20     -- the size of the episode/batch of sequenced timeseries we'll feed the rnn in the training
local predictionBatchSize = 20  -- the size of the batch of sequenced timeseries we'll feed the rnn in the validating
local lr = 0.02        -- the learning rate to apply to the weights


--[[ build up model definition ]]--

-- create a trend guessing model, "tmodel"
tmodel = nn.Sequential()                                               -- wrapping it all in Sequential brings forward / backward methods to all layers in one go
tmodel:add(nn.Sequencer(nn.Identity()))                                -- untransfomed input data

tmodel:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize, rho )))     -- will create a complex network of lstm cells that learns back rho timesteps
tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))                      -- I'm sticking in a place to do dropout, not strictly needed I don't think

tmodel:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize2, rho)))      -- creating a second layer of lstm cells
tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))
 
--tmodel:add(nn.Sequencer(nn.Linear(hiddenSize2, outputSize)))           -- reduce the output back down to the output class nodes
tmodel:add(nn.Sequencer(nn.SoftMax()))                             -- apply the log soft max, as we're guessing classes.
                                                                      -- when used with criterion of ClassNLLCriterion, is effectively CrossEntropy
-- set criterion
                                                                    -- arrays indexed at 1, so target class to be like [1,2], [2,1] not [0,1],[1,0]
criterion = nn.SequencerCriterion(nn.MSECriterion())              -- if you want a regression, use mse
--criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())           -- if you want a classifier, use this. 
-- criterion = nn.CrossEntropyCriterion()                           -- potential alt for LogSoftMax / ClassNLLCriterion, but you pass it 1d weights (??)



--[[ set out learning functions ]]--
function gradientUpgrade(model, x, y, criterion, learningRate, i)
  model:zeroGradParameters()
  local prediction = model:forward(x)
  local err = criterion:forward(prediction, y)
  if i % 10 == 0 then
    print('error for iteration ' .. i  .. ' is ' .. err/rho)
  end
  local gradOutputs = criterion:backward(prediction, y)
  if err > 0 then
  model:backward(x, gradOutputs)
  model:updateParameters(learningRate)
  model:zeroGradParameters()
  end
end


--[[ FEED THE NETWORK WITH VALUES ]]--

-- initialise some of the file counters
local offset = 0
local batchcounter = 0
local episode = 0
local userCounter = 0  
local id = "XYZ"

feed_input = {}     -- create a table to hold our input data we'll use to predict, I want a table of tensors
feed_output = {}    -- create a table to hold our target outputs to train against

-- Read data from file to tensor
--local file = io.open("test.csv", 'r')  
local file = io.open("data_for_network5.tsv", 'r') 

for line in file:lines('*l') do  
    
    local row = line:split('\t')

    -- Update number of access
    if id ~= row[4] then
      id = row[4]
      userCounter = 0
    end

    local tfidf = row[10]:split(';')

    if row[10] ~= "Nil" then
       userCounter = userCounter + 1
       offset = offset + 1
       batchcounter = batchcounter + 1
       data = torch.Tensor(inputSize) 

      for key, val in ipairs(tfidf) do
        if key == 1 then
          data[key] = userCounter -- Number of access
          data[key+1],data[key+2] = please_get_sin_and_cos(row[1]) -- Time in the sinus format and the cosinus fromat
        else
          data[key+2] = tonumber(val) -- Tfidf value
        end  
      end

      if offset % batchSize ~= 0 then
        feed_input[batchcounter] = data
      end
      if batchcounter ~= 1 then
        feed_output[batchcounter-1] = data:narrow(1,4,outputSize)
      end

    

      --[[ TRAIN THE RNN ]]-- (note we are still inside the file iterator here)
    
      if offset < threshold and offset % batchSize == 0 then    -- bactSize defines the episode size
        episode = episode + 1                               -- create a counter that tracks the total number of episodes we've created
         
        -- now send this batch episode to rnn backprop through time learning
        
        gradientUpgrade(tmodel, feed_input, feed_output, criterion, lr, episode)
        
        -- now clear out the tables, reset them     
        feed_input = nil; feed_input = {}
        feed_output = nil; feed_output = {}

        -- reset the rowID of the batch episode back to zero
        batchcounter = 1
        feed_input[batchcounter] = data
      end

      --[[ Validation ]]--
        
      if offset > threshold and offset % batchSize == 0 then        -- we have now rolled through the timeseries learning, but can we guess the ending?   
                                                                    -- note we are still inside the file iterator. 
       -- TEST OF PREDICTION --            
       -- grab the current row of inputs, and generate prediction
        
        -- initialise Tensors for validation
        local desired_output = torch.Tensor(batchSize,527)
        local given_output = torch.Tensor(batchSize,527)

        local prediction_output = tmodel:forward(feed_input)
        
        for i=1,batchSize do
          given_output[i] = prediction_output[i]
          desired_output[i] = feed_output[i]
        end
        
        print("Predicated output: ")
        print(prediction_output[1])

        print("Desirable output: ")
        print(feed_output[1])
      
        -- T-SNE reduciton dimensionality of vectors
        local d = m.embedding.tsne(desired_output, {dim=2, perplexity=30})
        local p = m.embedding.tsne(given_output, {dim=2, perplexity=30})
        
        print('T-SNE for desired outputs:')
        print(d)

        print('T-SNE for predicated outputs:')
        print(p)

        gnuplot.plot{
         {'original',d:squeeze(),'+'},
         {'prediction',p:squeeze(),'+'}
        }
        gnuplot.axis('equal')
        gnuplot.axis{-20,50,-10,30}
        gnuplot.grid(true)
        local answer
        repeat
           io.write("for continue operation press y")
           io.flush()
           answer=io.read()
        until answer=="y"

      end     
    end
end

