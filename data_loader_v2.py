import re
import pandas as pd
import numpy as np
import tensorflow as tf

pattern = re.compile(r'\d+')


def session(graph=None, allow_soft_placement=False, log_device_placement=False,
            allow_growth=True):
    """return a session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


class TXTLoader(object):
    def __init__(self, root, txt_path, batch_size, height=650, width=800,
                 transformer_fn=None, num_epochs=None, shuffle=True, min_after_dequeue=25,
                 allow_smaller_final_batch=False, num_threads=2, seed=None):
        root = root.replace('\\', '/')
        root = root if root[-1] == '/' else '{}/'.format(root)
        self.root = root
        self.txt_path = txt_path.replace('\\', '/')
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.transformer_fn = transformer_fn
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.min_after_dequeue = min_after_dequeue
        self.allow_smaller_final_batch = allow_smaller_final_batch
        self.num_threads = num_threads
        self.seed = seed

        df = pd.read_csv(self.txt_path, header=None, sep=';')

        if shuffle:
            df = df.sample(frac=1., random_state=seed)

        thumbnail_dims_df = df.apply(lambda line: list(map(int, pattern.findall(line[1]))), axis=1)
        bboxes_df = df.apply(lambda line: list(map(int, pattern.findall(line[2]))), axis=1)

        images_path = df[0].values
        thumbnail_dims = thumbnail_dims_df.values
        bboxes = bboxes_df.values

        u, indices, inverse_indices, bboxes_counts = np.unique(images_path, return_index=True, return_inverse=True,
                                                               return_counts=True)
        sorted_indices = np.sort(indices)
        images_path_array = images_path[sorted_indices]

        self.n_sample = len(images_path_array)

        bboxes_counts_array = bboxes_counts[inverse_indices][sorted_indices]
        max_bboxes_count = np.max(bboxes_counts)
        splits = np.cumsum(bboxes_counts_array)
        thumbnail_dims_array = np.zeros((self.n_sample, max_bboxes_count, 2))
        bboxes_array = np.zeros((self.n_sample, max_bboxes_count, 4))
        tdss = np.split(thumbnail_dims, splits)
        bbss = np.split(bboxes, splits)
        for index, (ip, bb_cnt, td, bb) in enumerate(zip(images_path_array, bboxes_counts_array, tdss, bbss)):
            thumbnail_dims_array[index, :bb_cnt, :] = np.array(list(td))
            bboxes_array[index, :bb_cnt, :] = np.array(list(bb))

        print('{}: create session!'.format(self.__class__.__name__))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                names = tf.convert_to_tensor(images_path_array, tf.string)
                images_path = tf.convert_to_tensor(list(map(lambda x: '{}{}'.format(root, x), images_path_array)),
                                                   tf.string)
                thumbnail_dims = tf.convert_to_tensor(thumbnail_dims_array, tf.int64)
                bboxes_count = tf.convert_to_tensor(bboxes_counts_array, tf.int64)
                bboxes = tf.convert_to_tensor(bboxes_array, tf.int64)
                image_path, thumbnail_dim, bbox, bbox_count, name = tf.train.slice_input_producer(
                    [images_path, thumbnail_dims, bboxes, bboxes_count, names],
                    shuffle=shuffle,
                    capacity=self.n_sample,
                    seed=seed,
                    num_epochs=num_epochs)

                image_value = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image_value, channels=3)
                shape = tf.shape(image)
                h, w = shape[0], shape[1]
                top_pad = (height - h) // 2
                bottom_pad = height - h - top_pad
                left_pad = (width - w) // 2
                right_pad = width - w - left_pad
                padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
                image = tf.pad(image, padding, mode='constant', constant_values=0)
                window = tf.convert_to_tensor([top_pad, left_pad, h + top_pad, w + left_pad])
                image = tf.reshape(image, [height, width, 3])
                shift = tf.cast(tf.convert_to_tensor([left_pad, top_pad, left_pad, top_pad]), dtype=tf.int64)
                gt_bbox = tf.add(bbox, shift)

                # image = tf.image.resize_images(image, [height, width])

                if shuffle:
                    capacity = min_after_dequeue + (num_threads + 1) * batch_size
                    image_batch, thumbnail_dim_batch, bbox_batch, gt_bbox_batch, bbox_count_batch, shape_batch, window_batch, name_batch = tf.train.shuffle_batch(
                        [image, thumbnail_dim, bbox, gt_bbox, bbox_count, shape, window, name],
                        batch_size=batch_size,
                        capacity=capacity,
                        min_after_dequeue=min_after_dequeue,
                        num_threads=num_threads,
                        allow_smaller_final_batch=allow_smaller_final_batch,
                        seed=seed)
                else:
                    capacity = (num_threads + 1) * batch_size
                    image_batch, thumbnail_dim_batch, bbox_batch, gt_bbox_batch, bbox_count_batch, shape_batch, window_batch, name_batch = tf.train.batch(
                        [image, thumbnail_dim, bbox, gt_bbox, bbox_count, shape, window, name],
                        batch_size=batch_size,
                        capacity=capacity,
                        allow_smaller_final_batch=allow_smaller_final_batch)
                self.batch_ops = [image_batch, thumbnail_dim_batch, bbox_batch, gt_bbox_batch, bbox_count_batch,
                                  shape_batch,
                                  window_batch, name_batch]
                if num_epochs is not None:
                    self.init = tf.local_variables_initializer()
        self.sess = session(graph=self.graph)
        if num_epochs is not None:
            self.sess.run(self.init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self.n_sample

    def batch(self):
        return self.sess.run(self.batch_ops)

    def __del__(self):
        print('{}: stop threads and close session!'.format(self.__class__.__name__))
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


if __name__ == '__main__':
    dataloader = TXTLoader(root='/Users/aiyoj/Downloads/Thumbnail Data Set/PQ_Set',
                           txt_path='./data/train_set.txt',
                           batch_size=1,
                           shuffle=False)
    print(len(dataloader))
    for step in range(10):
        batch = dataloader.batch()
        print(batch[0].shape, batch[1][0], batch[2][0], batch[3][0], batch[4][0], batch[5][0], batch[6][0], batch[7][0])
