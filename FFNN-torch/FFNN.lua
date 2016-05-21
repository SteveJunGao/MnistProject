-- do all the things in this file
-- my first nerual network program written by torch

require 'torch'
require 'xlua'
require 'nn'
require 'image'
require 'optim'

train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'

-- I use only a fraction of the data to train
-- which might run faster
-- original number is 73257 & 26032
train_size = 73200
test_size = 26000

print '==> loading dataset'

loaded = torch.load(train_file, 'ascii')
train_data = {
	-- TODO why need a dot here
	-- maybe the grammer 
	-- cause if i omit the dot it failed
	data = loaded.X:transpose(3, 4),
	labels = loaded.y[1],
	-- the original file is a function
	size = function() return train_size end
}

loaded = torch.load(test_file, 'ascii')
test_data = {
	data = loaded.X:transpose(3, 4),
	labels = loaded.y[1],
	size = function() return test_size end
}

print '==>  preprocessing data'

train_data.data = train_data.data:float()
test_data.data = test_data.data:float()

print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,train_data:size() do
   train_data.data[i] = image.rgb2yuv(train_data.data[i])
end
for i = 1,test_data:size() do
   test_data.data[i] = image.rgb2yuv(test_data.data[i])
end


channels = {'y','u','v'}

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = train_data.data[{ {},i,{},{} }]:mean()
   std[i] = train_data.data[{ {},i,{},{} }]:std()
   train_data.data[{ {},i,{},{} }]:add(-mean[i])
   train_data.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   test_data.data[{ {},i,{},{} }]:add(-mean[i])
   test_data.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,train_data:size() do
      train_data.data[{ i,{c},{},{} }] = normalization:forward(train_data.data[{ i,{c},{},{} }])
   end
   for i = 1,test_data:size() do
      test_data.data[{ i,{c},{},{} }] = normalization:forward(test_data.data[{ i,{c},{},{} }])
   end
end


noutputs = 10
nfeats = 3
width = 32
height = 32
ninputs = nfeats * width * height

nhiddens = 1500

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, nhiddens))
-- TODO maybe we don't need this layer
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens, noutputs))


model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print(model)
print(criterion)

print '==> defining some tools'
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
confusion = optim.ConfusionMatrix(classes)

-- we temporary don't use the log file
train_log = optim.Logger(paths.concat('log', 'train.log'))
test_log = optim.Logger(paths.concat('log', 'test.log'))
 
if model then 
	parameters, grad_parameters = model:getParameters()
end

learning_rate = 1e-3
bach_size = 1

-- optimization is Stochastic Gradient Decent
optim_state = {
	learningRate = learning_rate,
	weightDecay = 0,
	momentum = 0,
	learningRateDecay = 1e-7
}
optim_method = optim.sgd

print '==> defining training procedure'

function train()
	-- body
	epoch = epoch or 1
	local time = sys.clock()
	model:training()
	shuffle = torch.randperm(train_size)
	print '==> doing epoch on training data'
	for t = 1, train_data:size(), bach_size do
		xlua.progress(t, train_data.size())
		local inputs = {}
		local targets = {}
		for i = t, math.min(t + bach_size - 1, train_data:size()) do
			
			local input = train_data.data[shuffle[i]]
			local target = train_data.labels[shuffle[i]]
			input = input:double()
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		local f_eval = function (x)
			if x ~= parameters then
				parameters:copy(x)
			end
			grad_parameters:zero()
			local f = 0
			for i = 1,#inputs do
				local output = model:forward(inputs[i])
				local err = criterion:forward(output, targets[i])
				f = f + err
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], df_do)
				confusion:add(output, targets[i])
			end
			grad_parameters:div(#inputs)
			f = f / #inputs
			return f, grad_parameters
		end

		optim_method(f_eval, parameters, optim_state)
	end
	time = sys.clock() - time
	time = time / train_data:size()
	print("\nepoch = ",epoch)
	print ("\n==> time to learn 1 sample = " ..(time * 1000).. 'ms')
	print(confusion)
	train_log:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	train_log:style{['% mean class accuracy (train set)'] = '-'}
	train_log:plot()
	confusion:zero()
	epoch = epoch + 1
end

function test()
	local time = sys.clock()
	model:evaluate()
	print '==> testing on the test set'
	for t = 1, test_data:size() do
		xlua.progress(t, test_data:size())
		local input = test_data.data[t]
		input = input:double()
		local target = test_data.labels[t]
		local prediction = model:forward(input)
		confusion:add(prediction, target)
	end
	time = sys.clock() - time
	time = time / test_data:size()
	print("\n==> time to test 1 sample = "..(time * 1000).. 'ms')
	print(confusion)
	test_log:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	test_log:style{['% mean class accuracy (test set)'] = '-'}
	test_log:plot()
	confusion:zero()
	-- body
end

for i = 1, 200 do 
	train()
	test()
end
