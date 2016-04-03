require 'torch'
require 'nn'
require 'time'
require 'RNN'
require 'SLSTM'
require 'optim'
  

-- tfidf vector has dimensionality of 527 
-- Number of all 1858699 
-- 834038
--10319
local VecTfidfSize = 527
local threshold = 10319 -- train up to this offset in the file, then predict.
local inputSize = 545      -- the number features in the csv file to use as inputs 
local hiddenSize = 300    -- the number of hidden nodes
local hiddenSize2 = 150   -- the second hidden layer
local outputSize = 1      -- the number of outputs representing the prediction targat. 
local dropoutProb = 0.4   -- a dropout probability, might help with generalisation
local rho = 4           -- the timestep history we'll recurse back in time when we apply gradient learning
local batchSize = 1     -- the size of the episode/batch of sequenced timeseries we'll feed the rnn in the training
local predictionBatchSize = 10  -- the size of the batch of sequenced timeseries we'll feed the rnn in the validating
local lr = 0.1       -- the learning rate to apply to the weights
local epoch = 30

local error_in_train = 0
local validation = false    

sgd_params = {
   learningRate = 0.01,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0.4
}

--[[ build up model definition ]]--

-- create a trend guessing model, "tmodel"
--tmodel = nn.Sequential()                                               -- wrapping it all in Sequential brings forward / backward methods to all layers in one go
--tmodel:add(nn.Sequencer(nn.Identity()))                                -- untransfomed input data

--tmodel:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize))) 
--tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))                      -- I'm sticking in a place to do dropout, not strictly needed I don't think

--tmodel:add(nn.Sequencer()
  --nn.FastLSTM(inputSize, hiddenSize, rho )))     -- will create a complex network of lstm cells that learns back rho timesteps
--tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))                      -- I'm sticking in a place to do dropout, not strictly needed I don't think
--tmodel:add(nn.Sequencer(nn.FastLSTM(hiddenSize, 1, rho)))      -- creating a second layer of lstm cells


tmodel = nn.Sequencer(
  nn.Sequential()
    :add(nn.Linear(inputSize,hiddenSize))
    --:add(nn.FastLSTM(inputSize, hiddenSize, rho))
    --:add(nn.Linear(hiddenSize,1))https://camo.githubusercontent.com/3ea758e7796a3e21d6b002f7aa588361d7e0bb7b/687474703a2f2f64336b62707a626d63796e6e6d782e636c6f756466726f6e742e6e65742f77702d636f6e74656e742f75706c6f6164732f323031352f31302f53637265656e2d53686f742d323031352d31302d32332d61742d31302e33362e35312d414d2e706e67
    :add(nn.Dropout(dropoutProb))
    :add(nn.FastLSTM(hiddenSize, 150, rho))
    :add(nn.Dropout(dropoutProb))
    :add(nn.Linear(150,10))
    :add(nn.LogSoftMax())
    --:add(nn.Linear(300,1))
    --:add(nn.Tanh())
  )

-- set criterion
                                                                    -- arrays indexed at 1, so target class to be like [1,2], [2,1] not [0,1],[1,0]
--criterion = nn.SequencerCriterion(nn.MSECriterion())              -- if you want a regression, use mse
--criterion = nn.SequencerCriterion(nn.AbsCriterion())             -- if you want a regression, use mse
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

--- TO-DO
-- ZMENIT CAS NA binarne hodnoty
-- vyskusat ine architektury


local path_to_model = "models/task3/model14.dat"
local model_to_save = "models/task3/model14.dat"

local write_method = "w"

if validation == true then
  path_to_data = "sacbee_day/new_validation_data_reduced.txt"
  threshold = 0
  epoch = 1
  tmodel = torch.load(path_to_model)
else
  path_to_data = "sacbee_day/new_train_data_reduced.txt"
  epoch = 1
  threshold = 435846
end

tmodel:remember("both") 

-- Data for train error
local file_for_train_error = io.open("results_files/task3/train_error_for14.txt", write_method)

-- Files for test error
local file_for_test_error = io.open("results_files/task3/test_error14.txt", write_method)

-- Other files
local file_for_predication = io.open("results_files/task3/data_for_visulaziation_of_predicated14.txt", write_method)
local file_for_target = io.open("results_files/task3/data_for_visulaziation_of_target14.txt", write_method)

local file_for_first_tip_error = io.open("results_files/task3/first_tip_error14.txt",write_method)

--local time_and_date_fil = io.open("results_files/task3/time_and_date.txt", "w")

function scale_number(number,from,to)
  -- Scale number from interval:
  local a = from[1]
  local b = from[2]
  -- To interval:
  local c = to[1]
  local d = to[2]

 -- if number > b then
   -- number = b
  --end

  local scale_number = (((number - a)*(d - c))/(b-a))+c
  return scale_number
end



function do_something(num,size)
  local data = torch.Tensor(size):zero()
  if num == nil or num == 0 then
    num = size
  end  

  data[num] = 1
  return data
end


function get_class(num,end_number)
  local num = tonumber(num)

  return do_something(num,10)
end 

function  count_accuracy(confusion_metrix,episodes)
  print(confusion_metrix)
  local numColumn = confusion_metrix:size(2)
  local value = 0
  
  for i=1, numColumn do
    value = confusion_metrix[i][i] + value
  end
  return value/episodes
end



function format_string(str)
  str = string.gsub(str,"'","")
  str = string.gsub(str,'"',"")

  return str
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

--[[ set out learning functions ]]--
function gradientUpgrade(model, x, y, criterion, learningRate, episode, offset)
  model:zeroGradParameters()

  local prediction = model:forward(x)
  
  local err = criterion:forward(prediction, y)
  local gradOutputs = criterion:backward(prediction, y)
  
  error_in_train = err + error_in_train

  if err > 0 then
    model:backward(x, gradOutputs)
    model:updateParameters(learningRate)
    model:zeroGradParameters()
  end

  if episode % 1000 == 0 then
    --print(error_in_train/offset)
    --file_for_train_error:write(tostring(error_in_train/offset).."\n")
    
    --print("Offset je:")
    --print(offset)
    --torch.save(path_to_model, tmodel)
  end

end

function write_to_the_confusion_metrix(confusion_metrix,prediction_output,feed_output)
  local y, i = torch.max(prediction_output[1], 1)
  local x = feed_output[1]           
  y = i[1]
  confusion_metrix[x][y] =  confusion_metrix[x][y] + 1
  return confusion_metrix    
end

function get_data_for_article(row)
  data = torch.Tensor(VecTfidfSize) 
  
  for i=1,VecTfidfSize do
    data[i] = tonumber(row[i+7]) -- Tfidf value
  end

  m = nn.SoftMax()
  data = m:forward(data)

  data:resize(inputSize)

  return data
end

function build_input_data(row, creation_date, first_occurence, data,vis)
  
  local mask_data = get_class(tonumber(row[535]))
  for i=0,9 do
    data[inputSize-(7+i)] = mask_data[i+1] 
  end  

  local days_from_creation = get_time_in_days(creation_date,row[5])
  days_from_creation = scale_number(days_from_creation,{1,400},{0,1})
  data[inputSize-6] = days_from_creation
  --print(number_of_days_from_creation)

  local hours_from_first_access = get_time_in_hours(first_occurence[1],row[5],first_occurence[2],row[6])
  hours_from_first_access = scale_number(hours_from_first_access,{1,1000},{-1,1})
  data[inputSize-5] = hours_from_first_access

  -- Number of access per hour
  --print(vis)
  --local visits = vis + tonumber(row[7])
  local number_of_access_per_hour = scale_number(tonumber(row[7]),{1,1000},{-1,1}) 

  data[inputSize-4] = number_of_access_per_hour

  local time_output = get_time_for_date_of_acess(row[5])
  
  data[inputSize-3] = time_output[1]        
  --data[inputSize-5] = time_output[2]
  data[inputSize-2] = time_output[3]
  --data[inputSize-3] = time_output[4]

  local time_output = get_time_for_time_of_acess(row[6])
  data[inputSize-1] = time_output[1]
 -- data[inputSize-1] = time_output[2]

  -- Flag for a new user
  data[inputSize] = 0

  --print(number_of_access_per_hour)

  local number_for_check_of_predicition = torch.Tensor(1)
 -- number_for_check_of_predicition[1] = scale_number(tonumber(row[7]),{1,1000},{-1,1})
  number_for_check_of_predicition[1] = tonumber(row[535])

  return data, number_for_check_of_predicition, visits
end

x, dl_dx = tmodel:getParameters()

--[[ FEED THE NETWORK WITH VALUES ]]--
for e = 1, epoch do

  -- initialise some of the file counters
  local confusion_metrix = torch.Tensor(10,10):zero()

  error_in_train = 0
  error_in_test = 0
  error_for_first = 0
  local visits = 0
  local first_counter = 0
  local offset = 0
  local batchcounter = 0
  local episode = 0 
  local article_id = 0
  local episode_for_test = 0  
  local first_occurence = {}
  local creation_date = 0
  local flag_for_first = false
  feed_input = {}     -- create a table to hold our input data we'll use to predict, I want a table of tensors
  feed_output = {}    -- create a table to hold our target outputs to train against

  -- Read data from file to tensor

local file = io.open(path_to_data, 'r') 


  tmodel:training()
  --local line = file:read("*l")

  for line in file:lines('*l') do  
    offset = offset + 1

    if offset > 0 then
      line = format_string(line)
      local row = line:split(';')

      
      if article_id ~= row[1] then
        visits = 0
        first_occurence = {row[5],row[6]}
        data = get_data_for_article(row)

        if row[3] ~= "NA" then
          creation_date = row[3]
        else
          creation_date = row[5]
        end
      end  

      if offset > threshold then 
        --time_and_date_file:write(tostring(row[5])..";"..tostring(row[6]).."\n")
      end

      batchcounter = batchcounter + 1


      local data, data_for_output,vis = build_input_data(row, creation_date, first_occurence, data:clone(),visits)
      visits = vis
      
      feed_input[batchcounter] = data
    
      if batchcounter ~= 1 then
        feed_output[batchcounter-1] = data_for_output
      end

      if article_id ~= row[1] then 
        feed_input[batchcounter-1] = nil
        feed_output[batchcounter-1] = nil
      end

     -- if article_id ~= row[1] or row[536] == "TRUE" then   
      if article_id ~= row[1] or validation == false then             
        feed_input[batchcounter] = nil

        --[[ TRAIN THE RNN ]]-- (note we are still inside the file iterator here)
        if offset < threshold and feed_input[1] ~= nil then    -- bactSize defines the episode size
          episode = episode + 1       -- create a counter that tracks the total number of episodes we've created
        
       
        local func = function(x_new)
          --tmodel:zeroGradParameters()

          if x ~= x_new then
            x:copy(x_new)
          end

          local small_feed_input = subrange(feed_input,1,table.getn(feed_input)-1)
          tmodel:forward(small_feed_input)
          
          local last_input = feed_input[table.getn(feed_input)]
          local last_output = feed_output[table.getn(feed_output)]
          local prediction = tmodel:forward({last_input})
          
          dl_dx:zero()
          local err = criterion:forward(prediction, {last_output})
          local gradOutputs = criterion:backward(prediction, {last_output})
          tmodel:backward({last_input}, gradOutputs)

          return err, dl_dx
        end

        --print(dl_dx)
        local _,fs  = optim.sgd(func,x,sgd_params)

        error_in_train = fs[1] + error_in_train
        -- now send this batch episode to rnn backprop through time learning
         -- gradientUpgrade(tmodel, feed_input, feed_output, criterion, lr, episode, offset)
        end

        --[[ Validation ]]--
        if offset > threshold then
                        -- we have now rolled through the timeseries learning, but can we guess the ending?  
          if episode_for_test == 0 then
            tmodel:evaluate()
            --tmodel:forget()
            --criterion = nn.SequencerCriterion(nn.MSECriterion())
          end

          episode_for_test = episode_for_test + 1                     -- note we are still inside the file iterator. 
         -- TEST OF PREDICTION --            
         -- grab the current row of inputs, and generate prediction
          
          -- initialise Tensors for validation'
          local prediction_output = tmodel:forward(feed_input)
          local err = criterion:forward(prediction_output, feed_output)

          error_in_test = err + error_in_test
          --print(error_in_test / offset)

          if flag_for_first == true then
            local err_for_first = nn.ClassNLLCriterion():forward(prediction_output[1], feed_output[1])
            error_for_first = err_for_first + error_for_first
            first_counter = first_counter + 1
            flag_for_first = false

          end

          for i,input in ipairs(prediction_output) do             
            local y, f  = torch.max(input, 1) 
            local x = feed_output[i][1]   
            y = f[1]        

            file_for_target:write(tostring(x).."\n")
            file_for_predication:write(tostring(y).."\n")

            confusion_metrix[x][y] = confusion_metrix[x][y] + 1
          end

        end

        -- now clear out the tables, reset them     
          feed_input = nil; feed_input = {}
          feed_output = nil; feed_output = {}
       
        if article_id ~= row[1] then
          feed_input = nil; feed_input = {}
          --print("Eine Grose Katastrophe")
          if offset > threshold then
            file_for_target:write("New Article"..","..tostring(row[1]).."\n")
            file_for_predication:write("New Article"..","..tostring(row[1]).."\n")
            --time_and_date_file:write("New Article"..","..tostring(row[1]).."\n")
          end

          local input = torch.Tensor(inputSize):fill(0)
          input[inputSize] = 1
          feed_input[1] = input
          tmodel:forward(feed_input)
          tmodel:forget()

          batchcounter = 2
          local first_data = data:clone() 

          first_data[inputSize-4] = scale_number(tonumber(0),{1,500},{-1,1}) 
          feed_input[1] = first_data
          feed_output[1] = data_for_output:clone()
          flag_for_first = true
        else
          batchcounter = 1
        end

          -- reset the rowID of the batch episode back to zero
          article_id = row[1]
          feed_input[batchcounter] = data:clone()

      end
    end
  end
  

  print("The End of the Epoch")
  print(episode)
  print(error_in_train/episode)
  file_for_train_error:write(string.gsub(tostring(error_in_train/episode),"%.",",").."\n")
    
  print(error_in_test/episode_for_test)
  file_for_test_error:write(string.gsub(tostring(error_in_test/offset),"%.",",").."\n")  
  print(error_for_first/first_counter)
  file_for_first_tip_error:write(string.gsub(tostring(error_for_first/first_counter),"%.",",").."\n")

  print(count_accuracy(confusion_metrix,episode_for_test))
  
  if validation ~= true then

    tmodel:forget()
    torch.save(model_to_save, tmodel)
  end

end  

file_for_first_tip_error:close()
file_for_test_error:close()
file_for_train_error:close()
file_for_predication:close()
file_for_target:close()
