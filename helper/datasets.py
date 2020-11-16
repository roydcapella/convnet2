#Dataset helpers

def retrieve_dataset(data_dir, num_classes, num_threads, shuffle_size, batch_size):
  if num_threads > 0:
    tfr_train_file=[os.path.join(data_dir, "train_{}.tfrecords".format(idx)) for idx in range(num_threads)]
    tfr_test_file=[os.path.join(data_dir, "test_{}.tfrecords".format(idx)) for idx in range(num_threads)]      
  else:
    tfr_train_file = os.path.join(data_dir, "train.tfrecords")
    tfr_test_file = os.path.join(data_dir, "test.tfrecords")
  
  sys.stdout.flush()

  mean_file = os.path.join(data_dir, "mean.dat")
  shape_file = os.path.join(data_dir,"shape.dat")

  input_shape =  np.fromfile(shape_file, dtype=np.int32)
  mean_image = np.fromfile(mean_file, dtype=np.float32)
  mean_image = np.reshape(mean_image, input_shape)

  #loading tfrecords into dataset object
  #train
  tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
  tr_dataset = tr_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, num_classes, with_augmentation = True));    
  tr_dataset = tr_dataset.shuffle(shuffle_size)        
  tr_dataset = tr_dataset.batch(batch_size = batch_size)    
  #'test'
  val_dataset = tf.data.TFRecordDataset(tfr_test_file)
  val_dataset = val_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, num_classes, with_augmentation = False));    
  val_dataset = val_dataset.batch(batch_size = batch_size)

  #define the model input
  input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')
  return  tr_dataset, val_dataset, input_image, input_shape, mean_image

def get_classes_from(mapping_file):
  items=[]
  assert os.path.exists(mapping_file)        
  with open(mapping_file, "r") as ins:
      arr = ins.read().split("\n") 
      for line in arr:
        classe=line.split("\t")[0]
        if classe !='':
          items.append(classe)  
  return items

def get_images_from(sample_file, classes, size):
  assert os.path.exists(sample_file)        
  # reading data from files, line by line
  filenames = []
  with open(sample_file) as file :        
    lines = [line.rstrip() for line in file]     
    random.shuffle(lines)
    _lines = [tuple(line.rstrip().split('\t'))  for line in lines ] [:size]
    filemap, labels = zip(*_lines)
    labels=np.array(labels).astype('int').tolist()
    for index, item in enumerate(labels):
      filenames.append([item, classes[item], filemap[index]])
  return filenames

def transform_dataset(dataset, name = "", save = False):
  elements =[]
  for (img, img_label) in dataset:
      elements.append([np.argmax(img_label), img])
  if save == True:
    with open(name +".txt", 'wb') as pyfile:  
      pickle.dump(elements, pyfile)
  return elements

def load_dataset_transformed_from_disk(name):
  validation = pickle.load(open(name +"_validation.txt", "rb" ))
  training = pickle.load(open(name +"_training.txt", "rb" ))
  return validation, training