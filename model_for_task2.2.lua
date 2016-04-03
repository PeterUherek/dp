require 'torch'
require 'nn'
require 'time'
require 'RNN'
require 'SLSTM'
require 'optim'

local threshold = 0  -- train up to this offset in the file, then predict.
local VecTfidfSize = 527
--local inputSize = 38
local inputSize = 47       -- the number features in the csv file to use as inputs 
local hiddenSize = 800     -- the number of hidden nodes
local hiddenSize2 = 527     -- the second hidden layer
local outputSize = 527      -- the number of outputs representing the prediction targat. 
local dropoutProb = 0.4   -- a dropout probability, might help with generalisation
local rho = 10            -- the timestep history we'll recurse back in time when we apply gradient learning
local batchSize = 1     -- the size of the episode/batch of sequenced timeseries we'll feed the rnn in the training
local predictionBatchSize = 10  -- the size of the batch of sequenced timeseries we'll feed the rnn in the validating
local lr = 0.01        -- the learning rate to apply to the weights
local epoch = 1 

local error_in_train = 0
local validation = false      

--[[ build up model definition ]]--

tmodel = nn.Sequencer(
  nn.Sequential()
    :add(nn.Linear(inputSize,300))
    --:add(nn.FastLSTM(inputSize, hiddenSize, rho))
    :add(nn.Dropout(dropoutProb))
    :add(nn.FastLSTM(300, 200, rho))
    --:add(nn.Dropout(dropoutProb))
   -- :add(nn.FastLSTM(300, 200, rho))
    :add(nn.Dropout(dropoutProb))
    :add(nn.Linear(200,2))
    :add(nn.LogSoftMax())
  )
                                                                  
--criterion = nn.SequencerCriterion(nn.DistKLDivCriterion())
--criterion = nn.SequencerCriterion(nn.MarginCriterion())
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

--Data for train error

local path_to_model = "models/task2/modelZ1.1.forget.dat"
local model_to_save = "models/task2/modelZ1.1.forget.dat"
local write_method = "w"
local path_to_validation_data = "data_for_task2/validation_dataZ7.txt"
local path_to_data = "data_for_task2/train_dataZ7.txt"

if validation == true then
  --path_to_data = "data_for_task2/validation_dataY1.txt"
  --threshold = 0
  epoch = 1
  tmodel = torch.load(path_to_model)
else
  --path_to_data = "data_for_task2/train_dataY1.txt"
  epoch = 16
  --threshold = 170000
  --tmodel = torch.load(path_to_model)
end

tmodel:remember("both") 

--Files for test error
local file_for_train_error = io.open("results_files/task2/train_error3.txt", write_method)
local file_for_test_error = io.open("results_files/task2/test_error3.txt", write_method)

function load_section_tables(path)
  local section_file = io.open(path,"r")
  local section_table = {}
  local counter = 1

  for line in section_file:lines('*l') do
  section_table[line] = counter
  counter = counter + 1  
  end

  section_file:close()
  return section_table
end

local section1_table = load_section_tables("section2.txt")

function change_output(prediction_output)
  local m = nn.Sequential()
  m:add(nn.Sequencer(nn.Exp()))      
  prediction_output=m:forward(prediction_output)
  return prediction_output
end

function format_string(str) 
  str = string.gsub(str,"'","")
  str = string.gsub(str,'"',"")

  return str
end

function scale_number(number,from,to)
  if number == nil then
    return 0
  end

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

function substract_section_for_articles(str,end_number)
  local str = str:split('/')
  local num = section1_table[str[4]]

  return do_something(num,15)
end 

function substract_section_for_device(num,end_number)
  local num = tonumber(num)

  return do_something(num,9)
end 


function get_feed_output(str)
  local feed_output = {}

  if str == "N" then
    feed_output = torch.Tensor{1}
  else
    feed_output = torch.Tensor{2}
  end
  return feed_output
end

function get_statement(str)
  local num = nil

  if str == "TRUE" then
    num = 1
  else
    num = 0
  end
  return num
end


function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

--[[ set out learning functions ]]--
function gradientUpgrade(model, x, y, criterion, learningRate, i, offset)
  model:zeroGradParameters()

  local prediction = model:forward(x)
  --print(y[1])
  --print(x[1])
  local err = criterion:forward(prediction, y)
  error_in_train = error_in_train + err

  local gradOutputs = criterion:backward(prediction, y)
  --print(prediction[1])
  if err > 0 then
    model:backward(x, gradOutputs)
    model:updateParameters(learningRate)
    model:zeroGradParameters()
  end

  if i % 100000 == 0 then
 print(error_in_train/i)
  --file_for_train_error:write(tostring(error_in_train/offset).."\n")
    --print("Episoda je:")
  print(i)
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


function build_data(row)

  local data = torch.Tensor(inputSize) 

  if row[2] == "117" then
    data[inputSize-45] = 1
  else
    data[inputSize-45] = 0
  end
  
  local mask_data = do_something(num,7)
  for i=0,6 do
    data[inputSize-(38+i)] = mask_data[i+1] 
  end 

  data[inputSize-37] = is_weekend(row[5])

  local mask_data = substract_section_for_articles(row[1])
  for i=0,14 do
    data[inputSize-(22+i)] = mask_data[i+1] 
  end  

  local mask_data = substract_section_for_device(row[16])
  for i=0,8 do
    data[inputSize-(14+i)] = mask_data[i+1] 
  end  

  data[inputSize-13] = get_statement(row[17])

  data[inputSize-12] = scale_number(tonumber(row[15]),{0,1},{0,1})

  data[inputSize-11] = scale_number(tonumber(row[14]),{0,26556},{-1,1})

  data[inputSize-10] = scale_number(tonumber(row[13]),{0,1780},{-1,1})

  data[inputSize-9] = scale_number(tonumber(row[12]),{0,734},{-1,1})

  data[inputSize-8] = scale_number(tonumber(row[9]),{0,9},{0,1})

  data[inputSize-7] = get_statement(row[11])

  data[inputSize-6] = get_statement(row[10])

  data[inputSize-5] = get_statement(row[8])

  data[inputSize-4] = get_statement(row[7])

  local time_output = please_get_time(row[5])
  data[inputSize-3] = time_output[1]        
  --data[inputSize-5] = time_output[2]
  data[inputSize-2] = time_output[3]
  --data[inputSize-3] = time_output[4]
  data[inputSize-1] = time_output[5]
  --data[inputSize-1] = time_output[6]

  data[inputSize] = 0
  
  return data
end


--[[ FEED THE NETWORK WITH VALUES ]]--
for e = 1, epoch do

local file = nil

if e%2 == 0 then
  file = io.open(path_to_validation_data, 'r') 
  threshold = 0
  validation = true
else
  file = io.open(path_to_data, 'r') 
  validation = false
  threshold = 400000
end

-- initialise some of the file counters
error_in_train = 0
error_in_test = 0

local confusion_metrix = torch.Tensor(2,2):zero()
local offset = 0
local batchcounter = 0
local episode = 0
local user_id = 0
local article_id = 0
local episode_for_test = 0  
local one_user = 0
feed_input = {}     -- create a table to hold our input data we'll use to predict, I want a table of tensors
feed_output = {}    -- create a table to hold our target outputs to train against


-- Read data from file to tensor
--local file = io.open(path_to_data, 'r') 

tmodel:training()

  for line in file:lines('*l') do
    line = format_string(line)
    offset = offset + 1
    if offset > 0 then
      local row = line:split(',')

      if offset == 1 then
        user_id = row[3]
        feed_output = get_feed_output(row[4])
      end  

      if article_id ~= row[2] or user_id ~= row[3] then
        batchcounter = batchcounter + 1

        article_id = row[2]

        local data = build_data(row)
        feed_input[batchcounter] = data

        --[[ TRAIN THE RNN ]]-- (note we are still inside the file iterator here)
        if user_id ~= row[3]  then              
          feed_input[batchcounter] = nil

          if offset < threshold then    -- bactSize defines the episode size
            episode = episode + 1       -- create a counter that tracks the total number of episodes we've created
            
          -- now send this batch episode to rnn backprop through time learning
            local small_feed_input = subrange(feed_input,1,table.getn(feed_input)-1)
            tmodel:forward(small_feed_input)

            local last_input = feed_input[table.getn(feed_input)]

            gradientUpgrade(tmodel, {last_input}, {feed_output}, criterion, lr, episode, offset)
              --gradientUpgrade(tmodel, last_input, feed_output, criterion, lr, episode, offset)

          end

          --[[ Validation ]]--
          if offset > threshold then
            if episode_for_test == 0 then
              tmodel:evaluate()
              tmodel:forget()
              --criterion = nn.SequencerCriterion(nn.MSECriterion())
            end
                                                                         -- we have now rolled through the timeseries learning, but can we guess the ending?   
            episode_for_test = episode_for_test + 1                     -- note we are still inside the file iterator. 
           -- TEST OF PREDICTION --            
           -- grab the current row of inputs, and generate prediction

            local small_feed_input = subrange(feed_input,1,table.getn(feed_input)-1)
            
            tmodel:forward(small_feed_input)

            local last_input = feed_input[table.getn(feed_input)]

           -- print(feed_output)
            local prediction_output = tmodel:forward({last_input})
            local err = criterion:forward(prediction_output, {feed_output})

            --prediction_output = change_output(prediction_output)     
            --print_results(batchSize,prediction_output,feed_output,err)
          
            confusion_metrix = write_to_the_confusion_metrix(confusion_metrix,prediction_output,feed_output)
      
            error_in_test = err + error_in_test
          end

          if user_id ~= row[3] then
            feed_input = nil; feed_input = {}

            local input = torch.Tensor(inputSize):fill(0)
            input[inputSize] = 1
            feed_input[1] = input
            --tmodel:forward(feed_input)
            tmodel:forget()
          end
          -- now clear out the tables, reset them     
          feed_input = nil; feed_input = {}
          feed_output = nil; feed_output = {}

          -- reset the rowID of the batch episode back to zero
          batchcounter = 1
          feed_input[batchcounter] = data:clone()
          user_id = row[3]
          feed_output = get_feed_output(row[4])

        end
      end
    end
  end

  if validation ~= true then
    print(episode)
    print(error_in_train/episode)
    file_for_train_error:write(string.gsub(tostring(error_in_train/episode),"%.",",").."\n")

    tmodel:forget()
    torch.save(model_to_save, tmodel)
  else
    print(episode_for_test)
    tmodel:forget()
    print(error_in_test/episode_for_test)
    file_for_test_error:write(string.gsub(tostring(error_in_test/episode_for_test),"%.",",").."\n")  

    print(confusion_metrix)
    local accuracy = ((confusion_metrix[1][1]+confusion_metrix[2][2])/episode_for_test)
    print(accuracy)
    local precision = confusion_metrix[2][2]/(confusion_metrix[2][2]+confusion_metrix[1][2])
    print(precision)
    local recall = confusion_metrix[2][2]/(confusion_metrix[2][2]+confusion_metrix[2][1])
    print(recall)
    local false_negative_rate = 1-recall
    print(false_negative_rate)

  end
end  

file_for_test_error:close()
file_for_train_error:close()

