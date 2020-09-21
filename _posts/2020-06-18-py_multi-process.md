---

layout: post

title: 一个Python多线程问题

date:  2020-06-18 12:00

tags: 

- Python

- 多线程

author: Lixiang Ru

permalink: /2020-06-18-py_multi-process

mathjax: true

---
最近跑一个github开源代码碰到了一个bug，出现bug的环境是 Ubuntu 16.04.6 + Python 3.6.8 + Pytorch 1.1.0 + RTX 2080Ti + CUDA 10.0；由于 不确定bug的具体原因，我把环境的包都描述上了。bug的现象是当程序执行若干循环之后，就会陷入stuck，我用了两张GPU，stuck之后其中一张会利用率为100%，另一张为0%。每次出现stuck时循环的次数不一定，可能几十个，也可能是两三百个，所以可排除是输入的原因。

大致可以确定出现bug时的调用语句：
``` python
for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
    img_name = img_name[0]; label = label[0]
    print(img_name)
    img_path = voc12.data.get_img_path(img_name, args.voc12_root)
    orig_img = np.asarray(Image.open(img_path))
    orig_img_size = orig_img.shape[:2]

    def _work(i, img):
        with torch.no_grad():
            with torch.cuda.device(i%n_gpus):
                _, cam = model_replicas[i%n_gpus](img.cuda())
                cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                    #print(cam.shape)
                return cam

    thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)), batch_size=12, prefetch_size=0, processes=args.num_workers)

    cam_list = thread_pool.pop_results()
```
这是其中BatchThreader类的定义代码。
``` python
class BatchThreader:

    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=12):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())

    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(self.pool.apply_async(self.func, args))
            #time.sleep(0.02)

    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result), len(self.left_args_list))

    def pop_results(self):

        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn
```

根据其中一张卡会在stuck时100%的现象，我推测，stuck时可能是 无限陷入line 66的__start_works中，也就是在执行line 84的if语句时，判定to_fill一直不为0，（可能是to_fill来不及更新？），我尝试在__start_works最后加了暂停一点时间，也就是注释掉的那句，似乎已解决这个bug。
也是有够诡异的。:disappointed_relieved: