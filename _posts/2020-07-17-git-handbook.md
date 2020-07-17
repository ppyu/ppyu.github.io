---
layout: post
title: Git使用总结
categories: Blog
description: 版本控制工具Git的常见用法。
keywords: 博客, Git
---

## 1、配置git信息

`$ git config --global user.name "Your Name"`
`$ git config --global user.email "email@example.com"`

注意`git config`命令的`--global`参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址。

## 2、创建版本库

在需要创建的文件夹里输入 `git init` 指令。

## 3、添加文件到Git仓库

要初始化一个Git仓库，使用`git init`命令。
添加文件到Git仓库，分两步：
**第一步**，使用命令`git add <file>`，注意，可反复多次使用，添加多个文件；
**第二步**，使用命令 `git commit` 完成。

**第一步**是用`git add`把文件添加进去，实际上就是把文件修改添加到**暂存区**；
**第二步**是用`git commit` 提交更改，实际上就是把暂存区的所有内容提交到**当前分支(工作区)**。

## 4、查看历史提交记录

`git log`
`git log` 命令显示从最近到最远的提交日志。
`git reflog` 记录每一次指令
如果嫌输出信息太多，看得眼花缭乱的，可以加上`--pretty=oneline`参数。
`git log --pretty=oneline`
你看到的一大串类似3628164...882e1e0的是commit id（版本号），和SVN不一样，Git的commit id不是1，2，3……递增的数字，而是一个SHA1计算出来的一个非常大的数字，用十六进制表示。

## 5、版本回退

在Git中，用`HEAD`表示当前版本。上一个版本就是`HEAD^`，上上一个版本就是`HEAD^^`，当然往上100个版本写100个`^`比较容易数不过来，所以写成`HEAD~100`。

使用 `git reset` 进行版本回退。
`git reset --hard HEAD^`
`git reset --hard 3628164`版本号没必要写全，前几位就可以了，Git会自动去找。当然也不能只写前一两位，因为Git可能会找到多个版本号，就无法确定是哪一个了。

## 6、查看修改状态

`git diff` #是工作区(work dict)和暂存区(stage)的比较
`git diff --cached` #是暂存区(stage)和分支(master)的比较
`git status` #查看修改状态
用`git diff HEAD -- filename` 命令可以查看工作区和版本库里面最新版本的区别。

## 7、撤销修改

命令`git checkout -- readme.txt`意思就是，把`readme.txt`文件在工作区的修改全部撤销，这里有两种情况：
一种是`readme.txt`自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；
一种是`readme.txt`已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。
总之，就是让这个文件回到最近一次`git commit`或`git add`时的状态。

## 8、关联远程仓库

`git remote add origin git@gitlab.intra.XXXX.com:ppy/LearnGit.git`
添加后，远程库的名字就是`origin`，这是Git默认的叫法，也可以改成别的，但是`origin`这个名字一看就知道是远程库。

## 9、把本地库的内容推送到远程库中

 `git push -u origin master`
 由于远程库是空的，我们第一次推送`master`分支时，加上了`-u`参数，Git不但会把本地的`master`分支内容推送的远程新的`master`分支，还会把本地的`master`分支和远程的`master`分支关联起来，在以后的推送或者拉取时就可以简化命令。此后，每次本地提交后，只要有必要，就可以使用命令`git push origin master`推送最新修改。

## 10、创建分支

`$ git checkout -b dev`
`git checkout`命令加上`-b`参数表示创建并切换，相当于以下两条命令：
`$ git branch dev`
`$ git checkout dev`

## 11、删除分支

`git branch -d dev`

## 12、查看所有分支

`git branch`

## 13、合并某分支到当前分支

`git merge <name>`

## 14、仓库克隆

当你从远程仓库克隆时，实际上Git自动把本地的`master`分支和远程的`master`分支对应起来了，并且，远程仓库的默认名称是`origin`。
要查看远程库的信息，用`git remote`。或者用`git remote -v`显示更详细的信息。

## 15、使用Git进行多人协作

多人协作的工作模式通常是这样：
首先，可以试图用`git push origin branch-name`推送自己的修改；
如果推送失败，则因为远程分支比你的本地更新，需要先用`git pull`试图合并；
如果合并有冲突，则解决冲突，并在本地提交；
没有冲突或者解决掉冲突后，再用`git push origin branch-name`推送就能成功！
如果`git pull`提示 *“no tracking information”*，则说明本地分支和远程分支的链接关系没有创建，用命令`git branch --set-upstream branch-name origin/branch-name`。
这就是多人协作的工作模式，一旦熟悉了，就非常简单。

上述建立分支链接的命令已经过时，使用下面的命令建立本地与远程仓库的链接。
`git branch --set-upstream-to=origin/rented_branch_name local_branch_name`
