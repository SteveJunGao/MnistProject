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

train_size = 10000
test_size = 10000
input_units = 28 * 28
hidden_units = 200
output_units = 10
learning_rate = 1e-3
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
	num_images = 10000
	-- creat a tensor
	images = torch.Tensor(num_images,input_units):zero()
	for t = 1, num_images do
		xlua.progress(t, num_images)
		im = buf_file:readByte(input_units)
		if im:size() ~= 784 then
			print(im:size())
		end
		for i = 1, input_units do
			if im[i] > 1 then 
				images[t][i] = 1
			else images[t][i] = 0
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
	num_labels = 10000
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
model:add(nn.Reshape(input_units))
model:add(nn.Linear(input_units, hidden_units))
model:add(nn.Sigmoid())
model:add(nn.Linear(hidden_units, output_units))
-- TODO maybe LogSoftMax
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

function train()
	epoch = epoch or 1
	model:training()
	shuffle = torch.randperm(train_size)
	print "==> doing epoch on training data"
	for t = 1, train_size do
		xlua.progress(t, train_size)
		local table_inputs = {}
		local inputs = train_images[shuffle[t]]:contiguous()
		local target = train_labels[shuffle[t]]
		--print (inputs)
		inputs = inputs:double()
		table.insert(table_inputs, inputs)
		criterion:forward(model:forward(table_inputs[1]), target + 1)
		model:zeroGradParameters()
		model:backward(table_inputs[1], criterion:backward(model.output, target + 1))
		model:updateParameters(1e-3)
		
	end
	epoch = epoch + 1
end

function test()
	print "==> testing on the tset set"
	count_true = 0
	model:evaluate()
	for t = 1, test_size do
		xlua.progress(t, test_size)
		inputs = test_images[t]:contiguous()
		inputs = inputs:double()
		target = test_labels[t]
		prediction = model:forward(inputs)
		predict = 0
		max = -1000000
		for i = 1, 10 do
			if prediction[i] > max then 
				max = prediction[i]
				predict = i
			end
		end
		if predict == (target + 1) then
			count_true = count_true + 1
		end
	end
	print(count_true)

end

for i = 1, 100 do 
	train()
	test()
end

