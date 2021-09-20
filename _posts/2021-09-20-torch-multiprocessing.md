---

layout: post

title: 【Python】多线程AttributeError: Cant pickle local object 

date:  2021-09-20 10:00

tags: 

- Python

author: Lixiang Ru

permalink: /2021-09-20-torch-multiprocessing

mathjax: true

---

在使用pytorch的多线程模块时，遇到一个bug

``` shell
Traceback (most recent call last):
  File "test_irn_mp.py", line 154, in <module>
    train(cfg=cfg)
  File "test_irn_mp.py", line 113, in train
    multiprocessing.spawn(_validate, nprocs=n_gpus, args=(wetr, split_dataset, cfg), join=True)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/site-packages/torch/multiprocessing/spawn.py", line 199, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/site-packages/torch/multiprocessing/spawn.py", line 148, in start_processes
    process.start()
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/process.py", line 105, in start
    self._popen = self._Popen(self)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/data/users/rulixiang/miniconda3/envs/py36/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't pickle local object 'boolean_dispatch.<locals>.fn
```

比较令人头疼的是这个error的trace log只有短短的一句 `multiprocessing.spawn(_validate, nprocs=n_gpus, args=(wetr, split_dataset, cfg), join=True)`, 实在是摸不清哪里出现的问题，只好强行google `AttributeError: Can't pickle local object `，最后出来找到一个类似的经验：多进程获取返回值是需要序列化的，多进程的函数中使用其他的类，而这些类可能存在一些不能被序列化的对象。

检查我的代码，发现是在模型初始化时会将pooling函数的句柄作为参数，猜测是这里导致的

```python
if pooling=="gmp":
    self.pooling = F.adaptive_max_pool2d
elif pooling=="gap":
    self.pooling = F.adaptive_avg_pool2d
```

将这里注释掉，解决了 :sweat_smile::sweat_smile::sweat_smile: