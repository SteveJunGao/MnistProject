-- operating the original file 

require 'torch'
require 'xlua'
require 'nn'
require 'image'
require 'optim'
require 'io'

train_images_filename = 'train-images.idx3-ubyte'
train_label_filename = 'train-labels.idx1-ubyte'
test_images_filename = 't10k-images.idx3-ubyte'
test_label_filename = 't10k-labels.idx1-ubyte'

train_size = 60000
test_size = 10000
input_units = 28 * 28
hidden_units = 200
output_units = 10
learning_rate = 1e-3
cnn_states = {6, 12, 100}
filt_size = 5
pool_size = 2
function get_images( filename)
	-- body
	print ("==> loading images from ", filename)
	-- open a binary file

	buf_file = torch.DiskFile(filename, 'r')
	buf_file:binary()
	-- bigEndian mode
	buf_file:bigEndianEncoding()
	-- read the first 4 info.
	status = buf_file:readInt(4)
	print(status)
	num_images = status[2]
	-- creat a tensor
	images = torch.Tensor(num_images, 1, 28, 28):zero()
	for t = 1, num_images do
		xlua.progress(t, num_images)
		im = buf_file:readByte(input_units)
		cnt = 1
		for i = 1, 28 do
			for j =1, 28 do
				if im[cnt] >= 1 then
					images[t][1][i][j] = 1
				else images[t][1][i][j] = 0
				end
				cnt = cnt + 1
			end
		end

	end
	return images
end

function get_labels(filename)
	print ("==>loading labels from", filename)
	buf_file = torch.DiskFile(filename, 'r')
	buf_file:binary()
	buf_file:bigEndianEncoding()
	status = buf_file:readInt(2)
	print(status)
	num_labels = status[2]
	labels = torch.Tensor(num_labels)
	for t = 1, num_labels do
		xlua.progress(t, num_labels)
		label = buf_file:readByte()
		labels[t] = label
	end
	return labels
end

train_images = get_images(train_images_filename)
test_images = get_images(test_images_filename)
train_labels = get_labels(train_label_filename)
test_labels = get_labels(test_label_filename)

model = nn.Sequential()
-- stage 1
model:add(nn.SpatialConvolution(1, cnn_states[1], filt_size, filt_size))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(pool_size, pool_size, pool_size, pool_size))
--stage 2
model:add(nn.SpatialConvolution(cnn_states[1], cnn_states[2], filt_size, filt_size))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(pool_size, pool_size, pool_size, pool_size))
--stage 3
model:add(nn.View(cnn_states[2] * 4 * 4))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(cnn_states[2] * 4 * 4, cnn_states[3]))
model:add(nn.Sigmoid())
model:add(nn.Linear(cnn_states[3], output_units))

--loss function
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
-- CUDA
--model:cuda()
--criterion:cuda()

classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
confusion = optim.ConfusionMatrix(classes)
train_log = optim.Logger('log', 'train.log')
test_log = optim.Logger('log', 'test.log')

function train()
	epoch = epoch or 1
	model:training()
	shuffle = torch.randperm(train_size)
	print " ==> doing epoch on training data"
	for t = 1, train_size do
		xlua.progress(t, train_size)
		local inputs = train_images[shuffle[t]]
		local target = train_labels[shuffle[t]]
		--print (inputs)
		inputs = inputs:double()
		local output = model:forward(inputs)
		confusion:add(output, target + 1)
		criterion:forward(output, target + 1)
		model:zeroGradParameters()
		model:backward(inputs, criterion:backward(model.output, target + 1))
		model:updateParameters(1e-3)
		
	end
	
	print ("epoch = ", epoch)
	print (confusion)
	train_log:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	train_log:style{['% mean class accuracy (train set)'] = '-'}
	train_log:plot()
	confusion:zero()
	epoch = epoch + 1

end

function test()
	print " ==> testing on the tset set"
	count_true = 0
	model:evaluate()
	for t = 1, test_size do
		xlua.progress(t, test_size)
		inputs = test_images[t]
		inputs = inputs:double()
		target = test_labels[t]
		prediction = model:forward(inputs)
		confusion:add(prediction, target + 1)
	end
	print(confusion)
	test_log:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	test_log:style{['% mean class accuracy (test set)'] = '-'}
	test_log:plot()
	confusion:zero()
end

for i = 1, 100 do 
	train()
	test()
end

