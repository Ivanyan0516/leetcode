# Leetcode刷题

1. 替换空格：

   请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

   使用自带的replace函数进行替换

```python
class Solution:
    def replaceSpace(self, s: str) -> str:  #函数说明
        return s.replace(" ","%20");
```

2. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

用head遍历链表，用stack保存链表取值，用 stack[::-1]将值倒序输出

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        stack = []
        while head!=None:
            stack.append(head.val)
            head = head.next
        return stack[::-1]
```





## 动态规划

### 简单

1. 最大子序和

   给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

   子数组的和tempsum和子数组和的最大值res;若tempsum+num<num,则说明之前的tempsum为负，丢弃

```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        tempsum = 0
        res = nums[0]
        for num in nums:
            tempsum = max(tempsum+num,num)
            res = max(tempsum,res)
        return res
```

2.加一

给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。你可以假设除了整数 0 之外，这个整数不会以零开头。

将数组转为string，再将string转为int，对数字进行加一操作，再将数字转为list

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        num = int(''.join([str(s) for s in digits]))+1
        return [int(s) for s in str(num)]
```



3.区域和检索 - 数组不可变

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210509225727210.png" alt="image-20210509225727210" style="zoom:50%;" />

解法1：对每次索引，用index保存nums中索引的值，再进行求和：注意self.nums的用法

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums


    def sumRange(self, left: int, right: int) -> int:
        index = self.nums[left:right+1]
        res = sum(x for x in index)
        return res

```

解法2： 用sums保存前n项的和，则i到j项的和即为前j项的和减去前i项的和。更快  复杂度都为O(N)

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = [0]
        sums = self.sums
        for i in range(len(nums)):
            sums.append(sums[-1]+nums[i])


    def sumRange(self, left: int, right: int) -> int:
        sums = self.sums
        return sums[right+1] - sums[left]

```



4.爬楼梯 **leetcode70**

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

解法1：动态规划

爬到n阶的方法总数dp[n]等于爬到n-1阶的方法总数加上爬到n-2阶的方法总数的和。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {}
        dp[1] = 1
        dp[2] = 2
        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

5. 除数博弈 **leetcode1025**

爱丽丝和鲍勃一起玩游戏，他们轮流行动。爱丽丝先手开局。

最初，黑板上有一个数字 N 。在每个玩家的回合，玩家需要执行以下操作：

选出任一 x，满足 0 < x < N 且 N % x == 0 。
用 N - x 替换黑板上的数字 N 。
如果玩家无法执行这些操作，就会输掉游戏。

只有在爱丽丝在游戏中取得胜利时才返回 True，否则返回 False。假设两个玩家都以最佳状态参与游戏。

解法1：

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        if n%2 == 0:
            return True
        else:
            return False
```



若拿到奇数，则因子必为奇数，bob则一直拿偶数，只用减1，Alice则一直拿奇数。故奇数必输，偶数必赢。

解法2； DP

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        # dp[1] = False
        # dp[2] = True
        for i in range(1,n+1):
            for j in range(1,i):
                if (i%j == 0) and (not dp[i-j]):
                    dp[i] = True
        return dp[n]  
```

6. 买卖股票的最佳时期 **leetcode121**

   给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

   你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

   返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

   解法1： DP

   当前i时刻最大利润 ： profit[i] = max(prices[i] - min(price[:i]) ,profit[i-1])

   minprice = min(minprice,price[i])

   ```python
   class Solution:
       def maxProfit(self, prices: List[int]) -> int:
           n = len(prices)
           profit = [0]*n
           minprice = prices[0]
           for i in range(1,n):
               profit[i] = max(profit[i-1], prices[i] - minprice)
               minprice = min(minprice,prices[i])
           return profit[n-1]
   ```

   

   7.判读子序列 **leetcode392**

   给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

   字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

   解法1：

   使用**string.find(str,x,y)**找到在string中str的索引，(optional: x,y 为限制string的范围取子序列) 。若string中无str，则返回-1.  

   ```python
   class Solution:
       def isSubsequence(self, s: str, t: str) -> bool:
           flag = 1
           for str in s:
               index = t.find(str)
               t = t[index+1:]
               if index == -1:
                   flag = 0
           return flag==1
   ```

8. 使用最小花费爬楼梯 **leetcode746**

数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。

每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。

请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。

解法1：

计dp[i]为从第1阶爬到阶梯顶花费的最小体力；则知道 更新规则 ：dp[i] = min(dp[i+1],dp[i+2]) +cost[i]

```python
class Solution:
    def minCostClimbingStairs(self, cost):
        n = len(cost)
        dp = [0]*n   #dp[i]代表从第i阶爬到阶梯顶花费的最小体力
        dp[n-1] = cost[n-1]
        dp[n-2] = cost[n-2]
        for i in range(n-3,-1,-1):
            dp[i] = min(dp[i+1],dp[i+2]) + cost[i]
        return min(dp[0],dp[1])
```

9. 三步问题  **面试题08.01**

三步问题。有个小孩正在上楼梯，楼梯有n阶台阶，小孩一次可以上1阶、2阶或3阶。实现一种方法，计算小孩有多少种上楼梯的方式。结果可能很大，你需要对结果模1000000007。

解法1：DP

计f(n)为上n阶台阶的方式总数，则更新规则 ： f(n) = f(n-1) + f(n-2) +f(n-3)

注意：此时的%1000000007应在dp[i]的

```python
class Solution():
    def waysToStep(self,n):
        if n ==1:
            return 1
        if n==2:
            return 2
        if n==3:
            return 4
        dp = [0]*(n+1)
        dp[1],dp[2],dp[3] = 1,2,4
        for i in range(4,n+1):
            dp[i] = (dp[i-1]+dp[i-2]+dp[i-3])%1000000007
        return dp[n]
```

10. 按摩师 **面试题17.16**

```python
class Solution:
    def massage(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n   #dp[i]代表第i次预约时的总最大预约分钟数
        dp = [nums[0]] + [max(nums[0],nums[1])] + [0] * (n - 2)
        for i in range(2,n):
            dp[i] = max(dp[i-2]+nums[i],dp[i-1])
        return dp[n-1]
```

11.买卖股票的最佳价格2

给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

只要第i+1天的价格比第i天的价格高，就在第i天买入，第i+1天卖出。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        profit = 0
        for i in range(n-1):
            if(prices[i]<prices[i+1]):
                profit = profit + prices[i+1] - prices[i]
        return profit
```

12.最长连续递增序列

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n==1:
            return 1
        dp = [1]*n
        for i in range(1,n):
            if nums[i]>nums[i-1]:
                dp[i] = dp[i-1] + 1
        return max(dp)
```

13.猜数字大小

猜数字游戏的规则如下：

每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：

1：我选出的数字比你猜的数字小 pick < num
-1：我选出的数字比你猜的数字大 pick > num
0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num
返回我选出的数字。

```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:
class Solution:
    def guessNumber(self, n: int) -> int:
        l,r = 1,n
        while l<=r:
            mid = (l+r)//2
            if guess(mid) == 0:
                res = mid 
                break
            elif guess(mid) == -1:
                r = mid - 1     #二分法，mid比猜的数字大，更新最右端r为mid-1
            elif guess(mid) == 1:
                l = mid + 1
        return res
```





### 中等

1. 等差数列划分

数组 A 包含 N 个数，且索引从0开始。数组 A 的一个子数组划分为数组 (P, Q)，P 与 Q 是整数且满足 0<=P<Q<N 

如果满足以下条件，则称子数组(P, Q)为等差数组：

元素 A[P], A[p + 1], ..., A[Q - 1], A[Q] 是等差的。并且 P + 1 < Q 。

函数要返回数组 A 中所有为等差数组的子数组个数。

解法1：

dp[i]表示在nums[i]处等差数列的长度减2，长度为N的等差数列的子等差数列数为 1+2+。。。+N-2

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n<3:
            return 0
        dp = [0]*n
        for i in range(2,n):
            if(nums[i]-nums[i-1] == nums[i-1] - nums[i-2]):
                dp[i] = dp[i-1]+1
        return sum(dp)
```

2. 子集 **leetcode78**

给你一个整数数组 nums ，**数组中的元素 互不相同** 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

解法1：递归

注意返回一个空列表 为   **return [[]]**

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 0:
            return [[]]
        result = self.subsets(nums[1:])
        res = result + [[nums[0]] + s for s in result]
        return res
```

3. 子集 **leetcode90**

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

解法1：递归

注意：避免重复子集的写法，需要先对列表进行排序 nums.sort()

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()   ##nums = nums.sort() wrong!!Nonetype
        if len(nums) == 0:
            return [[]]
        sub = self.subsetsWithDup(nums[1:])  
        res = sub + [[nums[0]] + s for s in sub if [nums[0]]+s not in sub]
        return res
```

4. 最长递增子序列 **leetcode300**

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

解法1：DP 超出时间限制

ｆ（ｘ）为以ｎｕｍｓ［ｘ］结尾的最长子序列的长度，则更新规则为：

f(x) = max f(j) +1 if nums[j]<nums[i]  else 1

```python
class Solution:
    def lengthOfLIS(self, nums):
        n = len(nums)
        if n==1:
            return 1
        dp = [1]*n
        for i in range(1,n):
            sub = []
            for j in range(0,i):
                if nums[j]<nums[i]:
                    sub.append(dp[j])
                if len(sub)>0:
                    dp[i] = max(sub)+1
                else:
                    dp[i]=1
        return max(dp)
```

解法2：不同的比较方式

```python
class Solution:
    def lengthOfLIS(self, nums):
        n = len(nums)
        if n==1:
            return 1
        dp = [1]*n
        res = 1
        for i in range(1,n):
            for j in range(0,i):
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
            res = max(dp[i],res)
        return res
```

5. 最小路径和 **leetcode64**

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

解法1： DP

f(I,j)为到达grid[i],[j]花费的最小路径和，则更新规则为：

f(i,j) = min(f(i-1,j),f(i,j-1)) + grid[i],[j]

注意 二维list维度的获取方式  **numpy.array(list).shape**             **m = len(grid),n = len(grid[0])**

给m行n列 二维list :dp赋初值时，使用       ==dp = [[0]*n for _ in range(m)]==

```python
import numpy
class Solution():
    def minPathSum(self,grid):
        m = numpy.array(grid).shape[0] #行   m = len(grid)
        n = numpy.array(grid).shape[1] #列   n = len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1,n):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for j in range(1,m):
            dp[j][0] = grid[j][0] + dp[j-1][0]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1])+ grid[i][j]        
        return dp[-1][-1]
```

6. 不同路径 **leetcode62**

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

解法1：DP

到达ij处的方法总数 dp[i],[j] = dp[i-1],[j] + dp[i],[j-1]

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

7. 不同路径2

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

```python
class Solution():
    def uniquePathsWithObstacles(self,grid):
        m,n = len(grid),len(grid[0])
        k =0
        if grid[0][0] == 1:
            return 0
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            if  grid[i][0]:
                for k in range(i,m):
                    dp[k][0] = 0
                break
            else:
                dp[i][0] = 1 
        for j in range(n):
            if  grid[0][j]:
                for k in range(j,n):
                    dp[0][k] = 0
                break
            else:
                dp[0][j] = 1 
        for i in range(1,m):
            for j in range(1,n):
                if grid[i][j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        print(dp)
        return dp[-1][-1]
```

8. 完全平方数

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

f(n)为组成正整数n所需的最小平方数的数量，则

f(n) = min( f(n), f(n-i*i)+1 )

```python
import math
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [_ for _ in range(n+1)]
        rt = math.ceil(math.sqrt(n))
        for i in range(1,rt+1):
            for j in range(i*i,n+1):
                dp[j] = min(dp[j],dp[j-i*i]+1)
        return dp[-1]
```

9. 最大正方形

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

 即dp[i],[j]为第(i,j)个位置上，其左上能包含最大正方形的面积，则更新公式为：

dp[i],[j] = min(dp[i-1],[j], dp[i],[j-1], dp[i-1],[j-1] ) +1

初始化，矩阵第一行第一列的dp值均为矩阵的值。

注意输入为matrix: List[List[str]] ，先转为数字

==max(map(max,dp))== 求矩阵元素中的最大值

或者 import numpy as np, a = np.array(dp)  a.max()

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix) #row
        n = len(matrix[0]) #column
        list = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                list[i][j] = int(matrix[i][j])
        dp = [[0]*n for _ in range(m)]
        max = 0
        for i in range(n):
            dp[0][i] = list[0][i]
        for i in range(m):
            dp[i][0] = list[i][0]       
        for i in range(1,m):
            for j in range(1,n):
                if list[i][j]:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j-1],dp[i-1][j]) + 1
        for i in range(m):
            for j in range(n):
                if dp[i][j] > max:   #最好不要用函数名去当作变量名
                    max = dp[i][j]
        #print(dp)
        return max*max

```

9. 整数拆分

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

定义dp[i]为正整数i拆分为至少两个正整数的和的 正整数的乘积的最大值，则更新规则为：

dp[i] = max( dp[j] X (i-j) , j X (i-j) ,dp[i] )

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        if n==1:
            return 0
        if n==2:
            return 1
        dp[0] = 0
        dp[1] = 0
        dp[2] = 1
        for i in range(3,n+1):
            maxj = []
            for j in range(i):
                maxj2 = max(j,dp[j])    #### 重要
                maxj.append(maxj2 * (i-j))
                dp[i] = max(maxj)
        print(dp)
        return dp[-1]
```

**注意一定要比较  j与dp[j]** ，如3 = 1+2 dp[2] =2,如果只赋值 dp[j] X (i-j)会因为 dp[2]=1导致 dp[3] =1*1=1

解法2： 记忆化搜寻 （递归）

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        assert(n >= 2)
        self.memo = [-1 for i in range(n + 1)]
        return self.backtrack(n)

    def backtrack(self, n):
        if n == 1:
            return 1
        if self.memo[n] != -1:
            return self.memo[n]
        res = float('-inf')
        for i in range(1, n):
            res = max(res, i*(n-i), i*self.backtrack(n-i))
        self.memo[n] = res
        return res
```

注意，**也要比较iX(n-i),  i X self.backtrack(n-i)** 来判断是否还需要分割。

10. 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

dp[i] ，偷窃到第i家时偷窃到的最高金额，则：

==dp[i] = max(dp[i-1], dp[i-2] + nums[i])== : 考虑了隔两家偷盗的情况

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1:
            return nums[0]
        dp = [0]*n
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        for i in range(2,n):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        print(dp)
        return dp[-1]
```

11. 打家劫舍2

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

==对于环状的数组==

思路： 同10，**把环状的房屋nums环 拆成两个队列，一个是从0到n-1，另一个是从1到n，然后返回两个结果最大的。**

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1:
            return nums[0]
        def Maxmoney(nums):
            n = len(nums)
            if n==1:
                return nums[0]
            dp = [0]*n
            dp[0] = nums[0]
            dp[1] = max(nums[0],nums[1])
            for i in range(2,n):
                dp[i] = max(dp[i-1],dp[i-2]+nums[i])
            return dp[-1]
        max1 = Maxmoney(nums[0:n-1])
        max2 = Maxmoney(nums[1:n])
        return max(max1,max2)       
```

12. 丑数1（简单）

给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。

丑数 就是只包含质因数 2、3 和/或 5 的正整数。

对这个正整数不断整除2,3,5，若最后值为1，则这个正整数n的质因子只有2，3，5其中的数，为丑数.（丑数无负数）  注意返回False的语法**return False**

```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if(n == 0):
            return False
        while(n%2 == 0):
            n = n/2
        while(n%3 == 0):
            n = n/3
        while (n%5 == 0):
            n = n/5
        print(n)
        return n==1
```

13. 丑数2

给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数** 。

依次增加丑数的2，3，5倍中的最小值

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        res = [1]
        t1,t2,t3 = 0,0,0
        while(len(res)<n):
            res.append(min(res[t1]*2,res[t2]*3,res[t3]*5))
            if res[-1] == res[t1]*2:
                t1 = t1 + 1
            if res[-1] == res[t2]*3:
                t2 = t2 + 1
            if res[-1] == res[t3]*5:
                t3 = t3 + 1
        #print(res)
        return res[-1]          
```

14.三角形最小路径和

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

动态规划： 注意初始化和更新公式

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0]*n for _ in range(n)]
        dp[0][0] = triangle[0][0]
        if n==1:
            return triangle[0][0]
        for i in range(1,n):
            dp[i][0] = dp[i-1][0] + triangle[i][0]
        for i in range(1,n):
            dp[i][i] = dp[i-1][i-1] + triangle[i][i]
        for i in range(1,n):
            for j in range(1,len(triangle[i])-1):
                dp[i][j] = min(dp[i-1][j],dp[i-1][j-1]) + triangle[i][j]
        #print(dp)
        return min(dp[-1])
```

15. 零钱兑换

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

dp[i] = min(dp[i-coin]) + 1    for coin in coins

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0]*(amount+1)
        for i in range(1,amount+1):
            temp = float('inf')
            for c in coins:
                if i-c >=0:
                    temp = min(temp,dp[i-c]+1)
                dp[i] = temp
        if dp[amount]==float('inf'):
            return -1
        #print(dp)                      
        return dp[amount]
```

16. 零钱兑换2

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount+1)
        dp[0] = 1
        for c in coins:
            for i in range(c,amount+1):
                dp[i] += dp[i-c]
        return dp[-1]           
```



16. 最大乘积子集

给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

思路：因为有正有负，所以需要准备两个数组来保存到i位置前的最大最小值。

最大值更新为前一个位置的最大值、最小值与当前值的乘积 与当前值的最大值

 ```python
 class Solution:
     def maxProduct(self, nums: List[int]) -> int:
         n = len(nums)
         M = [0]*n  #到第i个位置的连续子数组的最大值
         m = [0]*n  #最小
         res = M[0] = m[0] = nums[0]
         for i in range(1,n):
             M[i] = max(M[i-1]*nums[i],nums[i],m[i-1]*nums[i])
             m[i] = min(M[i-1]*nums[i],nums[i],m[i-1]*nums[i])
             res = max(res,M[i])
         return res
 ```

17.买卖股票的最佳时期含冷冻期

关于买卖股票的通用解法：https://blog.csdn.net/qq_32424059/article/details/94559656

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0 for _ in range(2)] for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        if len(prices)<2:
            return 0
        for i in range(1,len(prices)):
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i])
            dp[i][1] = max(dp[i-1][1],dp[i-2][0]-prices[i]) #卖了要等一天的冷静期
            #dp[i][1] = max(dp[i-1][1],dp[i-1][0]-prices[i]) #无冷冻期
        print(dp)
        return dp[-1][0]
```

18.买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

在每次交易后利润费用减一个手续费即可。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0]*2 for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        if n == 1 :
            return 0
        for i in range(1,n):
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i]-fee)
            dp[i][1] = max(dp[i-1][1],dp[i-1][0]-prices[i])
        print(dp)
        return dp[-1][0]
```

19.计算各个位数不同的数字个数

给定一个**非负**整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10^n

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        dp = [0]*(n+1)
        if n == 0:
            return 1
        if n == 1:
            return 10
        dp[0],dp[1] = 1,10
        for i in range(2,n+1):
            m = 9
            for j in range(i-1):
                m = m*(9-j)
            dp[i] = m + dp[i-1]
        return dp[-1]
```

20.比特位计数

给定一个非负整数 **num**。对于 **0 ≤ i ≤ num** 范围中的每个数字 **i** ，计算其二进制数中的 1 的数目并将它们作为数组返回。

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        def count1(num):
            num_str = bin(num)[2:]  #ob101 == 5
            count = 0
            for str in num_str:
                if str == '1':
                    count = count + 1
            return count
        List = []
        for nums in range(num+1):
            List.append(count1(nums))
        return List
```

21.最大整除子集.（最长递增子序列）

给你一个由 无重复 正整数组成的集合 nums ，请你找出并返回其中最大的整除子集 answer ，子集中每一元素对 (answer[i], answer[j]) 都应当满足：
answer[i] % answer[j] == 0 ，或
answer[j] % answer[i] == 0
如果存在多个有效解子集，返回其中任何一个均可。

对于求方案数的题目，多开一个数组来记录状态从何转移而来是最常见的手段。然后利用回溯求原答案。

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        n = len(nums)
        nums.sort()
        f , g = [0]*n , [0]*n
        for i in range(n):
            length,prev = 1 , i
            for j in range(i):
                if nums[i]%nums[j] == 0:
                    if f[j] + 1 > length:
                        length = f[j] + 1
                        prev = j
            f[i] = length
            g[i] = prev
        max_len = idx = -1
        for i in range(n):
            if f[i]>max_len:
                idx = i
                max_len = f[i]
        res = []
        while(len(res)<max_len):
            res.append(nums[idx])
            idx = g[idx]
        res.reverse()
        return res
```

22. 最长递增子序列

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        dp = [0]*n
        dp[0] = 1
        for i in range(1,n):
            max_len = 1
            for j in range(i):
                if nums[j]<nums[i]:
                    if dp[j]+1 > max_len:
                        max_len = dp[j]+1
            dp[i] = max_len
        return max(dp)
```

23.最长递增子序列的个数

给定一个未排序的整数数组，找到最长递增子序列的个数。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        dp = [1]*n   #结尾为nums[i]的最长递增子序列的长度
        count = [1]*n  #结尾为nums[i]的最长递增子序列的个数
        for i in range(1,n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j]+1 > dp[i]:
                        dp[i] = dp[j]+1
                        count[i] = count[j] 
                    elif dp[j]+1 == dp[i]:
                        count[i] = count[i] + count[j]
        max_len = max(dp)
        # res = 0
        # for i in range(n):
        #     if dp[i] == max_len:
        #         res = res + count[i]
        # return res
        return sum(c for i,c in enumerate(count) if dp[i] == max_len)                     
```

24.一和零

给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        def findmn(str): 
            m,n = 0,0
            for s in str:
                if s=='0':
                    m = m + 1
                elif s == '1':
                    n = n + 1
            return m,n
        dp = [[0]*(n+1) for _ in range(m+1)]  #表示m和0和n个1 能组成的最大子集个数
        for str in strs:
            cost_zero,cost_one = findmn(str)
            for i in range(m,cost_zero-1,-1):   #需要倒序
                for j in range(n,cost_one-1,-1): 
                    dp[i][j] = max(dp[i][j],1 +dp[i-cost_zero][j-cost_one])
        return max(map(max,dp))
```

25.出界的路径数

给定一个 m × n 的网格和一个球。球的起始坐标为 (i,j) ，你可以将球移到相邻的单元格内，或者往上、下、左、右四个方向上移动使球穿过网格边界。但是，你最多可以移动 N 次。找出可以将球移出边界的路径数量。答案可能非常大，返回 结果 mod 109 + 7 的值。

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        N,i,j = maxMove,startRow,startColumn
        paths = [[[-1]*(N+1) for _ in range(n+1)] for _ in range(m+1)]
        def dfs(m,n,N,i,j):
            MOD = 1e9 + 7
            if (i<0 or j<0 or i>=m or j>=n):
                return 1
            if (N == 0):
                return 0
            if(paths[i][j][N] != -1):
                return paths[i][j][N]
            res = 0
            dirs = [[0,1],[0,-1],[1,0],[-1,0]]
            for dir in dirs:
                ni = i + dir[0]
                nj = j + dir[1]
                res += dfs(m,n,N-1,ni,nj)
            paths[i][j][N] = int(res%MOD)
            return paths[i][j][N]
        return dfs(m,n,N,i,j)
```

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        N,i,j = maxMove,startRow,startColumn
        paths = [[[0]*(N+1) for _ in range(n+1)] for _ in range(m+1)]
        dirs = [[0,1],[0,-1],[1,0],[-1,0]]
        res = 0
        Mod = 1000000007
        paths[i][j][0] = 1
        for k in range(1,N+1):
            for i in range(m):
                for j in range(n):
                    for dir in dirs:
                        ni = i + dir[0]
                        nj = j + dir[1]
                        if(ni<0 or nj<0 or ni>=m or nj>=n):
                            res = (res + paths[i][j][k-1])
                        else:
                            paths[ni][nj][k] = (paths[ni][nj][k] + paths[i][j][k-1])
                        res %= Mod
        return res
```

26.马在棋盘上的概率

已知一个 NxN 的国际象棋棋盘，棋盘的行号和列号都是从 0 开始。即最左上角的格子记为 (0, 0)，最右下角的记为 (N-1, N-1)。 

现有一个 “马”（也译作 “骑士”）位于 (r, c) ，并打算进行 K 次移动。 

如下图所示，国际象棋的 “马” 每一步先沿水平或垂直方向移动 2 个格子，然后向与之相垂直的方向再移动 1 个格子，共有 8 个可选的位置。

```python
class Solution:
    def knightProbability(self, N,K,r,c) :
        dp = [[[-1]*(K+1) for _ in range(N+1)] for _ in range(N+1)]

        def dfs(N,K,i,j):
            if (i<0 or j<0 or i>=N or j>=N):
                return 0
            if K==0:
                return 1
            if (dp[i][j][K] != -1):
                return dp[i][j][K]
            dirs = [[1,2],[2,1],[2,-1],[1,-2],[-1,-2],[-2,-1],[-2,1],[-1,2]]
            res = 0    
            for dir in dirs:
                ni = i + dir[0]
                nj = j + dir[1]
                res = res + dfs(N,K-1,ni,nj)
            dp[i][j][K] = res
            return res
        stay = dfs(N,K,r,c)
        return stay/(8**K)
```

27.删除并获得点数

给你一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

思路： 与打家劫舍类似，注意一个数组可能有重复数字，将重复数字相加当作这家人的财产即可。

```python
class Solution:
    def deleteAndEarn(self, nums):
        maxval = max(nums)
        number = [0]*(maxval+1)
        for num in nums:
            number[num] += num
        def Earn(number):
            n = len(number)
            dp = [0]*n
            dp[0],dp[1] = number[0],max(number[0],number[1])
            for i in range(2,n):
                dp[i] = max(dp[i-1],dp[i-2]+number[i])
            return dp[-1]
        res = Earn(number)
        return res
```

28.猜数字大小2

我们正在玩一个猜数游戏，游戏规则如下：

我从 1 到 n 之间选择一个数字，你来猜我选了哪个数字。

每次你猜错了，我都会告诉你，我选的数字比你的大了或者小了。

然而，当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。直到你猜到我选的数字，你才算赢得了这个游戏。

```python
class Solution:    #记忆化
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(n+1)]
        res = self.cost(dp,1,n)
        return res

    def cost(self,dp,l,r):
        if l>=r:
            return 0
        if dp[l][r]:
            return dp[l][r]
        dp[l][r] = min(i + max(self.cost(dp,l,i-1),self.cost(dp,i+1,r)) for i in range(l,r+1))
        return dp[l][r]
```

```python
class Solution:   #迭代 DP
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(n+1)]
        for l in range(n-1,0,-1):
            for r in range(l+1,n+1):
                dp[l][r] = min(i + max(dp[l][i-1],dp[i+1][r]) for i in range(l,r))
        return dp[1][n]
```





==背包问题== 416 494 322 518 377 

29.目标和

给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

**注意这里判断一个列表中有多少子集和为一个目标值的数目的算法**

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        zonghe = 0
        for num in nums:
            zonghe += num
        target = target + zonghe
        if target%2 == 1 or target//2 > zonghe:
            return 0
        target = target//2   ##判断nums中有多少个子集和为target
        dp = [0]*(target+1)
        dp[0] = 1
        for num in nums:  
            for j in range(target,num-1,-1):  #倒序
                dp[j] = dp[j] + dp[j-num]
        return dp[-1]
```

30.组合总和

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：所有数字（包括 target）都是正整数。解集不能包含重复的组合。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(candidates,target,start,size,path,res):
            if target<0:
                return  #返回
            if target==0:
                res.append(path)
                return
            for index in range(start,size):
                dfs(candidates,target-candidates[index],index,size,path+[candidates[index]],res)#使用path+[candidates[index]]只改变函数中的path值，不会每次都改变path的值
        size = len(candidates)
        path = []
        res = []
        dfs(candidates,target,0,size,path,res)
        return res
```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res=[]
        candidates.sort()
        self.candidates=candidates
        self.dfs(0, [], target)
        return self.res
    
    def dfs(self,start,path,target):
        if target==0:
            self.res.append(path)
            return
        for i in range(start,len(self.candidates)):
            if self.candidates[i]>target:break
            #if i>start and self.candidates[i]==self.candidates[i-1]:continue           
            self.dfs(i, path+[self.candidates[i]], target-self.candidates[i])
```

31.组合总和2

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明：所有数字（包括目标数）都是正整数。解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res=[]
        candidates.sort()
        self.candidates=candidates
        self.dfs(0, [], target)
        return self.res
    
    def dfs(self,start,path,target):
        if target==0:
            self.res.append(path)
            return
        for i in range(start,len(self.candidates)):
            if self.candidates[i]>target:break
            if i>start and self.candidates[i]==self.candidates[i-1]:continue  #若当前这个元素与上一个元素相同则跳过       
            self.dfs(i+1, path+[self.candidates[i]], target-self.candidates[i]) #i+1而非i，从下一个位置开始找
```

32.组合总和3

找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

说明：所有数字都是正整数。解集不能包含重复的组合。

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        candidates = [i for i in range(1,10)]
        def find(candidates,start,path,res,target,k):
            if target == 0:
                if len(path) == k:
                    res.append(path)
                return
            for i in range(start,len(candidates)):
                if candidates[i]>target:  break
                find(candidates,i+1,path+[candidates[i]],res,target-candidates[i],k)
        target = n
        res,path = [],[]
        find(candidates,0,path,res,n,k)
        return res      
```

33.组合总和4

给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

 ```python
 class Solution:  #这种方法会超时
     def combinationSum4(self, nums: List[int], target: int) -> int:
         def dfs(nums,target,start,res):
             if target == 0:
                 res = res + 1
                 return
             for i in range(start,len(nums)):
                 if nums[i]>target:  break
                 dfs(nums,target-nums[i],0,res)
         res = 0
         nums.sort()
         dfs(nums,target,0,res)
         return res
 ```

```python
class Solution:#动态规划，组合总数为i的组合数 = 组合总数为i-num再在末尾加个num构成的组合；考虑了排列顺序
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0]*target
        for i in range(1,target+1):
            for num in nums:
                if num<=i:
                    dp[i] += dp[i-num]
        return dp[-1] 
```

34.组合

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def find(nums,start,path,k,res):
            if len(path) == k:
                res.append(path)
            for i in range(start,len(nums)):
                find(nums,i+1,path+[nums[i]],k,res)
        path = []
        res = []
        nums = [i for i in range(1,n+1)]
        find(nums,0,path,k,res)
        return res
```

35.摆动序列

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

```python

class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        # c是用来记录前一个差值是下降还是上升的，默认0
        n, c, res = len(nums), 0, 1 
        if n < 2:
            return n
        for i in range(1, n):
            x = nums[i] - nums[i - 1]
            # 如果有差值才继续处理，相等直接就跳过不处理了
            if x:
                # <0代表有上升下降的交替，=0是初始情况的判断
                if x * c <= 0:
                    res += 1
                c = x
        return res
```

```python
import numpy as np
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums)==1:    return 1
        Diff = []
        for i in range(0,len(nums)-1):
            Diff.append(nums[i]-nums[i+1])
        c,res = 0,1
        for diff in Diff:
            if diff!=0:
                if diff*c<=0:
                    res += 1
                    c = np.sign(diff)   #np.sign(0)= 0，正数为1，负数为-1
        return res
```



### 困难

1. 不同路径3

在二维网格 grid 上，有 4 种类型的方格：

1 表示起始方格。且只有一个起始方格。
2 表示结束方格，且只有一个结束方格。
0 表示我们可以走过的空方格。
-1 表示我们无法跨越的障碍。
返回在四个方向（上、下、左、右）上行走时，从起始方格到结束方格的不同路径的数目。

每一个无障碍方格都要通过一次，但是一条路径中不能重复通过同一个方格。

```python
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        start,end,p = (),(),1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    start = (i,j)
                elif grid[i][j] == 2:
                    end = (i,j)
                elif grid[i][j] == 0:
                    p += 1       
        def path(x,y,p):
            if not (0<=x<m and 0<=y<n and grid[x][y]>=0):
                return 0
            if (end == (x,y) and p == 0 ):
                return 1
            grid[x][y] = -1
            res = path(x-1,y,p-1) + path(x+1,y,p-1) + path(x,y-1,p-1) +path(x,y+1,p-1)
            grid[x][y] = 0
            return res
        k1,k2 = start
        return path(k1,k2,p)
```

2.买卖股票的最佳时机IV

给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）

注：初始化让其他值为负无穷表示不合理取值的步骤很关键。

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        k = min(n//2,k)
        buy = [[0]*(k+1) for _ in range(n)]
        sell = [[0]*(k+1) for _ in range(n)]
        if n<2:
            return 0
        buy[0][0] = -prices[0]
        sell[0][0] = 0
        for i in range(1, k + 1):
            buy[0][i] = sell[0][i] = float("-inf")
        for i in range(1,n):
            buy[i][0] = max(buy[i-1][0], sell[i-1][0]-prices[i])
            for j in range(1,k+1):
                buy[i][j] = max(buy[i-1][j],sell[i-1][j]-prices[i])
                sell[i][j] = max(sell[i-1][j],buy[i-1][j-1]+prices[i])
        return max(sell[-1])      
```

3.鸡蛋掉落

给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。

已知存在楼层 f ，满足 0 <= f <= n ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。

每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 x 扔下（满足 1 <= x <= n）。如果鸡蛋碎了，你就不能再次使用它。如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中 重复使用 这枚鸡蛋。

```python
class Solution:
    def superEggDrop(self, K, N):
        dp = [[0 for _ in range(N + 1)] for _ in range(K + 1)]
         # dp[K][m] == N，也就是给你 K 个鸡蛋，测试 m 次，最坏情况下最多能测试 N 层楼。
        for i in range(1, K + 1):
            for step in range(1, N + 1):
                dp[i][step] = dp[i - 1][step - 1] + (dp[i][step - 1] + 1)
                if dp[K][step] >= N:
                    return step
        return 0
```

```python
class Solution:
    def superEggDrop(self, k,n):
        if n == 1:
            return 1
        f = [[0]*(k+1) for _ in range(n+1)]
        for i in range(1,k+1):
            f[1][i] = 1
        res = -1
        for i in range(2,n+1):
            for j in range(1,k+1):
                f[i][j] = f[i-1][j-1] + f[i-1][j] + 1
            if f[i][k] >= n:
                res = i
                break
        return res
```







### 真题

1. 可被三整除的最大和   **leetcode1262**

给你一个整数数组 nums，请你找出并返回能被三整除的元素最大和。

法一：

求总和，若总和模3余1，则减去最小的  min(一个模3余1的num,模3余2的两个最小num之和);

若总和模3余2，则减去最小的 min(一个模3余2的num,模3余1的两个最小num之和);

```python
class Solution:
    def maxSumDivThree(self, nums):
        Sums = sum(num for num in nums)
        S1 = []
        S2 = []
        t1,t2 = 0,0
        for num in nums:
            if num%3 == 1:
                S1.append(num)
            if num % 3 ==2:
                S2.append(num)
            

        if Sums%3 == 0:
            return Sums
        elif Sums%3 == 1:
            if len(S1): x1 = min(S1); t1 = Sums - x1
            if len(S2)>2: x2 = sum(sorted(S2)[:2]) ; t2 =Sums - x2
            return max(t1,t2)
        elif Sums%3 == 2:
            if len(S2): x2 = min(S2) ;t2 =Sums - x2
            if len(S1)>2: x1 = sum(sorted(S1)[:2]) ;t1 = Sums - x1
            return max(t1,t2)
```

法二：动态规划

f(i)表示模3余i的最大sum，更新规则： 

![image-20210510121402353](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210510121402353.png)

```python
class Solution:
    def maxSumDivThree(self,nums):
        dp = [0,0,0] #f(i)的值
        for num in nums:
            for i in dp[:]:
                dp[(i+num)%3] = max( i + num, dp[(i+num)%3])
        return dp[0]
```

2. 最长定差子序列   **leetcode1218**

   给你一个整数数组 arr 和一个整数 difference，请你找出并返回 arr 中最长等差子序列的长度，该子序列中相邻元素之间的差等于 difference 。

   子序列 是指在不改变其余元素顺序的情况下，通过删除一些元素或不删除任何元素而从 arr 派生出来的序列。

   解法1：动态规划

   计dp[i]为子序列末尾值为i的最长定差子序列的最大长度， 若计算dp[i]，则考虑dp[i-k]，若在i值前的序列中存在i-k(k为定差)，则dp[i] = dp[i-k]+1，否则dp[i]=1。

   ```python
   class Solution:
       def longestSubsequence(self, arr: List[int], difference: int) -> int:
           dp = {}
           for i in arr:
               if (i - difference) in dp:  #dp.keys() also work
                   dp[i] = dp[i-difference]+1
               else :
                   dp[i] = 1
           return max(dp.values()) 
   ```

   注意 ： **判断一个key是否在字典中，取字典中的最大value值的取法**

3. 买卖股票的最佳时机3

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
## 超出时间限制
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 1:
            return 0
        def Profit(prices):
            m = len(prices)
            if m < 2:
                return 0
            profit = [0]*m
            minprice = prices[0]
            for i in range(1,m):
                profit[i] = max(profit[i-1],prices[i]-minprice)
                minprice = min(minprice,prices[i])
            return profit[-1]
        res = 0
        for i in range(1,n):
            res1 = Profit(prices[:i])
            res2 = Profit(prices[i-1:n])
            res = max(res,res1+res2)
        return res
```

```python
class Solution:
    def maxProfit(self, prices):
        n = len(prices)
        if n < 2:
            return 0
        dp1 = [0 for _ in range(n)]
        dp2 = [0 for _ in range(n)]
        minval = prices[0]
        maxval = prices[-1]
        #前向   
        for i in range(1,n):
            dp1[i] = max(dp1[i-1], prices[i] - minval)
            minval = min(minval, prices[i])
        #后向    
        for i in range(n-2,-1,-1):
            dp2[i] = max(dp2[i+1], maxval - prices[i])
            maxval = max(maxval, prices[i])
        
        dp = [dp1[i] + dp2[i] for i in range(n)]
        return max(dp)
```

这里dp1[i]可看作前向到第i天的最大收益，后向dp2[i]可看作第i天到最后一天的最大收益。











## python runoob学习

变量类型：

数据类型：Numbers（数字）String（字符串） List（列表） Tuple（元组） Dictionary（字典）

元组用 **()** 标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。

**set()** 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。

循环语句：

break continue pass

Numbers: math\ cmath(复数）模块

ceil(x)： 向上取整   math.floor(4.8) = 4     floor():向下取整   round(x)  round(x,n)(四舍五入保留n位小数)           cmp(x,y) 

String :

string.count(str, beg=0, end=len(string)) :返回 str 在 string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数

string.find(str, beg=0, end=len(string)) :检测 str 是否包含在 string 中，如果 beg 和 end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1

string.replace(str1, str2,  num=string.count(str1)) :把 string 中的 str1 替换成 str2,如果 num 指定，则替换不超过 num 次.

string.split(str="", num=string.count(str)):以 str 为分隔符切片 string，如果 num 有指定值，则仅分隔 **num+1** 个子字符串

List:

list(seq)  将元组转换为列表

tuple(seq)  将列表转换为元组。

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210509221345637.png" alt="image-20210509221345637" style="zoom: 50%;" />

判断列表是否相等   

```python
import operator
operator.eq(a,b)
a==b
```





## leetcode顺序刷题

1.两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**index()函数**

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            if (target-nums[i]) in nums[i+1:]: 
                return [i,nums[i+1:].index(target-nums[i])+i+1]  #注意第二个数的索引
```

2.两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(l1.val + l2.val)
        #print(head)
        cur = head
        while l1.next or l2.next:
            l1 = l1.next if l1.next else ListNode()
            l2 = l2.next if l2.next else ListNode()
            cur.next = ListNode(l1.val + l2.val + cur.val // 10)
            cur.val = cur.val % 10
            cur = cur.next
        if cur.val >= 10:
            cur.next = ListNode(cur.val // 10)
            cur.val = cur.val % 10
        return head
```

3.无重复字符的最长子串  ==滑动窗口解法==

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**set()函数**创建一个无序不重复元素集    

```python
class Solution:    
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:return 0
        left = 0
        lookup = set()  #set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])    #set.remove() 一直减到重复的字符
                left += 1
                cur_len -= 1
            if cur_len > max_len:max_len = cur_len
            lookup.add(s[i])               #set.add()
        return max_len
```

4.寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1),len(nums2)
        length = m+n
        if length==0:   return
        num = nums1 + nums2
        num.sort()
        if length%2 == 1:
            res = num[length//2]
        else:   res = (num[int(length/2)] + num[int(length/2-1)])/2
        return res
```

5.最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

```python
class Solution:
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    def longestPalindrome(self, s: str) -> str:
        start,end = 0,0
        for i in range(len(s)):
            left1,right1 = self.expandAroundCenter(s,i,i)
            left2,right2 = self.expandAroundCenter(s,i,i+1)
            if right1 - left1 >end -start:
                start,end = left1,right1
            if right2 - left2>end - start:
                start,end = left2,right2
        return s[start:end+1]
```

6.Z字形变换

将一个给定字符串 `s` 根据给定的行数 `numRows` ，以从上往下、从左到右进行 Z 字形排列。

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows<2:   return s
        res = ["" for _ in range(numRows)]   #构建空字符串列表
        i = 0
        flag = -1
        for str in s:
            res[i] += str
            if i==0 or i==numRows-1:
                flag = -flag
            i = i + flag
        # result = ""
        # for i in range(numRows):
        #     result += res[i]
        # return result 
        return "".join(res)              # "".join()
```

7.整数反转

给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。

```python
class Solution:
    def reverse(self, x: int) -> int:
        x_inv = 0
        flag = 1 if x>0 else -1
        x = x if x>0 else -x
        if -2**31>x or 2**31-1 <x: return 0
        while(x):
            x_inv = x_inv*10 + x%10
            x = x//10
        if -2**31>x_inv or 2**31-1 <x_inv: return 0
        return x_inv*flag
```

8. 字符串转换整数 (atoi)

```python
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        res = []                       # 有效的数字字符存储
        flag = True                    # 默认值为True，一旦有整数字符出现，则标记为False
        
        numslist = ['0','1','2','3','4','5','6','7','8','9'] 
        for i in range(len(str)):
            if str[i] ==' ' and flag:  # 如果均为空格字符，且无非空字符出现继续
                continue
                
            if str[i] == '+' and flag: # 如果“+”字符第一次出现，则添加到列表中，标记修改为False并继续
                res.append(str[i])
                flag = False
                continue
                
            if str[i] == '-' and flag: # 如果“-”字符第一次出现，则添加到列表中，标记修改为False并继续
                res.append(str[i])
                flag = False
                continue
                
            if str[i] not in numslist: # 除过上述情况，如果字符不为数字，则直接退出迭代
                break
            else:
                res.append(str[i])     # 反之有数字出现,添加到列表中，并修改标记为False
                flag = False
                
        res = ''.join(res)             # 拼接字符串 
        if res == '-' or res=='' or res=='+':
            return 0
        else :
            res = int(res)
        
        if res>2**31-1:               # 特殊情况处理
            return 2**31-1
        if res<-2**31:
            return -2**31
        else:
            return res

```

















# leetcode刷题指南

https://github.com/youngyangyang04/leetcode-master

**数组-> 链表-> 哈希表->字符串->栈与队列->树->回溯->贪心**

# 一.数组篇

### 1.二分法

注意定义的区间为左闭右闭区间还是左闭右开区间， 用于判断left能否与right相等、区间端点的更新公式。

1.1 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        l,r = 0,n-1
        flag = 0
        mid = (l+r)//2
        while(l<=r):
            if nums[mid]>target:
                r = mid - 1
            elif nums[mid]<target:
                l = mid + 1
            elif nums[mid]==target:
                flag = 1
                return mid
                break
            mid = (l+r)//2
        if flag==0: return -1
```

1.2 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l,r = 0,len(nums)-1
        mid = (l+r)//2
        flag = 0
        while(l<=r):
            if nums[mid]<target:
                l = mid + 1
            elif nums[mid]>target:
                r = mid - 1
            elif nums[mid]==target:
                flag = 1
                return mid
                break
            mid = (l+r)//2
        if flag==0: return mid+1
```

1.3 在排序数组中查找元素的第一个和最后一个位置

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。  如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums)==0:    return [-1,-1]
        start = end = -1
        for i in range(0,len(nums)):
            if nums[i]==target: end = i
        if end==-1:   return [-1,-1]                
        for i in range(0,len(nums)):
            if nums[i]==target:
                start = i
                break
        return [start,end]          #时间复杂度 O(n)
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums)==0:    return [-1,-1]
        if nums[0]>target or nums[-1]<target:
            return [-1,-1]
        l,r = 0,len(nums)-1
        flag = 0
        mid = (l+r)//2
        while(l<=r):
            if nums[mid]<target:
                l = mid + 1
            elif nums[mid]>target:
                r = mid - 1
            elif nums[mid]==target:
                flag = 1
                l=r=mid
                while l-1>=0 and nums[l-1]==target: l=l-1
                while r+1<=len(nums)-1 and nums[r+1]==target:   r=r+1
                break
            mid = (l+r)//2
        return [l,r] if flag else [-1,-1]         #二分查找 时间复杂度O(logn)
        # if flag==0:  return[-1,-1]
        # else:   return [l,r]
```

1.4 x的平方根

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        l,r = 1,x
        if x==0 :   return 0
        while(l<=r):
            mid = (l+r)//2
            if mid**2<x:   l = mid 
            elif mid**2>x:   r = mid 
            elif mid**2==x:    return mid
            if r-l==1:    
                return l
                break
# x = int(input())   #读取数字输入
# q = Solution()
# print(q.mySqrt(x))
```

```python
class Solution:   ##找出平方小于等于x的最大整数
    def mySqrt(self, x: int) -> int:
        l,r,ans = 0,x,-1
        while(l<=r):
            mid = (l+r)//2
            if mid*mid<x:
                ans = mid
                l = mid + 1
            if mid*mid>x:   r = mid - 1
            if mid*mid == x:    
                ans = mid   
                break
        return ans
```

1.5有效的完全平方数

给定一个 正整数 `num` ，编写一个函数，如果 `num` 是一个完全平方数，则返回 `true` ，否则返回 `false` 。

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        l,r,flag = 0,num,0
        while(l<=r):
            mid = (l+r)//2
            if mid*mid<num: l = mid + 1
            elif mid*mid>num:   r=mid-1
            elif mid*mid==num: 
                flag = 1 
                break
        return True if flag else False
```

### 2.移除元素

给一个数组和一个值val，需要 **原地** 移除所有数值为val的元素，并返回移除后数组的新长度。**空间复杂度 O(1)**

**要知道数组的元素在内存地址中是连续的，不能单独删除数组中的某个元素，只能覆盖。**

思路：

暴力解法 ：两个for循环，一个遍历数组元素，一个循环更新数组； 时间复杂度O($n^2$)

双指针法（快慢指针法）：通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。 时间复杂度O(n)

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        fast = slow = 0
        for fast in range(0,len(nums)):
            if nums[fast]!= val:
                nums[slow]=nums[fast]
                slow += 1
        return slow
```



2.1 删除排序数组中的重复项

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

```python
class Solution:   ##快慢指针法 
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 1
        for fast in range(1,len(nums)):
            if nums[fast]!=nums[fast-1]:
                nums[slow]=nums[fast]
                slow += 1
        return slow
```

2.2 移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        fast = slow = num = 0
        for fast in range(len(nums)):   
            if nums[fast]!=0:
                nums[slow]=nums[fast]         ## 也可以在这里直接将0与非零元素交换位置
                slow += 1
            else:   
                num += 1
        for i in range(len(nums)-num,len(nums)):   #将后面元素赋值为0
            nums[i] = 0
```

2.3 比较含退格的字符串

给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

```python
class Solution:   #常规做法 时间空间复杂度均为O(M+N)
    def backspaceCompare(self, s: str, t: str) -> bool:
        def backspace(s):
            ret = []
            for char in s:
                if char!= '#':  ret.append(char)
                elif len(ret)>0:    ret.pop()
            return "".join(ret)
        return backspace(s)==backspace(t)
```

```python
class Solution:         # 空间复杂度O(1) 时间O(M+N)
    def backspaceCompare(self, s: str, t: str) -> bool:
        i,j = len(s)-1,len(t)-1
        skips = skipt = 0
        while i>=0 or j>=0:
            while i>=0:
                if s[i]=='#':
                    skips += 1
                    i -= 1
                elif skips>0:
                    i -= 1
                    skips -= 1
                else:   break
            while j>=0:
                if t[j]=='#':
                    skipt += 1
                    j -= 1
                elif skipt>0:
                    skipt -= 1
                    j -= 1
                else:   break
            if i >= 0 and j >= 0:
                if s[i] != t[j]:
                    return False
            elif i >= 0 or j >= 0:
                return False
            i -= 1
            j -= 1
        return True              
```

### 3.有序数组的平方

给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

```python
class Solution:  #sort()归并排序  时间O(nlogn) 空间O(logn)
    def sortedSquares(self, nums: List[int]) -> List[int]:
        newnums = [num*num for num in nums]
        newnums.sort()
        return newnums
```

```python
class Solution:  #双指针 	 时间O(n)
    def sortedSquares(self, nums: List[int]) -> List[int]:
        i,j,pos = 0,len(nums)-1,len(nums)-1
        ans = [0]*len(nums)
        while(i<=j):
            if nums[i]*nums[i]>nums[j]*nums[j]:
                ans[pos]=nums[i]*nums[i]
                i += 1
            else:
                ans[pos] = nums[j]*nums[j]
                j -= 1
            pos -= 1
        return ans
```

### 4.长度最小的子数组 滑动窗口

   ==滑动窗口==

给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0

暴力解法：两个for循环计算大于等于s的子序列的最小长度，时间复杂度O(n2),空间复杂度O(1)

**滑动窗口**：所谓滑动窗口，就是不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果。

```python
class Solution:   ##滑动窗口 时间复杂度O(n)：每个元素只操作两次（进出滑动窗） 空间复杂度O(1)
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        res = float("inf")
        Sum = index = 0
        for i in range(len(nums)):
            Sum += nums[i]
            while Sum>=target:
                res = min(res,i-index+1) 
                Sum -= nums[index]
                index += 1
        return res if res!=float("inf") else 0
```

有两个补充题未做

4.1

4.2

### 5.螺旋矩阵2

给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0]*n for _ in range(n)]
        left = up = 0
        right = down = n-1
        num = 1
        while(left<=right and up <= down):
            for i in range(left,right+1): #left to right
                matrix[up][i]=num
                num += 1
            up += 1
            for i in range(up,down+1):
                matrix[i][right] = num
                num += 1
            right -= 1
            for i in range(right,left-1,-1):
                matrix[down][i] = num
                num += 1
            down -= 1
            for i in range(down,up-1,-1):
                matrix[i][left] = num
                num += 1
            left += 1
        return matrix          
```

5.1螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m,n = len(matrix),len(matrix[0])
        L = []
        left = up = 0
        right,down = n-1,m-1
        while left<=right and up<=down:
            for i in range(left,right+1):
                L.append(matrix[up][i])
            up += 1
            for i in range(up,down+1):
                L.append(matrix[i][right])
            right -= 1
            if left<=right and up<=down:  #针对m和n不同的情况，需要同时满足
                for i in range(right,left-1,-1):
                    L.append(matrix[down][i])
                down -= 1
                for i in range(down,up-1,-1):
                    L.append(matrix[i][left])
                left += 1
        return L
```

5.2顺时针打印矩阵 与上题相同

### 总结篇

![image-20210614152030543](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614152030543.png)

# 二.链表篇

### 基础篇

链表是一种通过指针串联在一起的线性结构，每一个节点是又两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。

链接的入口点称为列表的头结点也就是head。

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614152447391.png" alt="image-20210614152447391" style="zoom: 25%;" />

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614152515215.png" alt="image-20210614152515215" style="zoom: 50%;" />

循环链表

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614152551869.png" alt="image-20210614152551869" style="zoom:25%;" />

存储方式

![image-20210614152724580](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614152724580.png)

操作：增删链表节点

可以看出链表的增添和删除都是O(1)操作，也不会影响到其他节点。

但是要注意，要是删除第五个节点，需要从头节点查找到第四个节点通过next指针进行删除操作，查找的时间复杂度是O(n)。

![image-20210614153630568](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210614153630568.png)

数组在定义的时候，长度就是固定的，如果想改动数组的长度，就需要重新定义一个新的数组。

链表的长度可以是不固定的，并且可以动态增删， 适合数据量不固定，频繁增删，较少查询的场景。

### 1.移除链表元素  虚拟头节点

给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy_head = ListNode(next=head)
        cur = dummy_head
        while(cur.next!=None):
            if cur.next.val == val: 
                cur.next = cur.next.next
            else:   cur = cur.next
        return dummy_head.next
```

### 2.设计链表

![image-20210617165052663](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210617165052663.png)

```python
class Node:        ##单链表  空间复杂度 O(1) 时间复杂度O(k) k为元素索引
    def __init__(self,val):
        self.next = None
        self.val = val

class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dummy_head = Node(0)
        self.count = 0




    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index<0 or index>=self.count:    return -1
        cur = self.dummy_head
        for i in range(index+1):   ##注意这里由于有虚拟头节点要先再加1
            cur = cur.next
        return cur.val



    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0,val)


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.count,val)


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index<0:
            index = 0
        elif index>self.count:  return

        self.count += 1
        add_node = Node(val)
        cur = self.dummy_head
        for _ in range(index):
            cur = cur.next
        # cur.next = add_node     #注意顺序
        add_node.next = cur.next
        cur.next = add_node
    



    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index<0 or index>=self.count: return
        self.count -= 1
        cur = self.dummy_head
        for _ in range(index):
            cur = cur.next
        cur.next = cur.next.next




# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

```python
class Node:            #双链表 空间复杂度O(1) 时间O(min(k,N-k)) k为元素索引，n为链表长度
    def __init__(self,val):
        self.next = None
        self.val = val
        self.prev = None

class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dummy_head = Node(0)
        self.dummy_tail = Node(0)
        self.count = 0
        self.dummy_head.next = self.dummy_tail
        self.dummy_tail.prev = self.dummy_head


    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index<0 or index>=self.count: return -1
        if index+1<self.count-index:
            cur = self.dummy_head
            for _ in range(index+1):
                cur = cur.next
        else:   
            cur = self.dummy_tail
            for _ in range(self.count-index):
                cur = cur.prev
        return cur.val

    
    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        pred,succ = self.dummy_head,self.dummy_head.next
        self.count += 1
        to_add = Node(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        pred,succ = self.dummy_tail.prev,self.dummy_tail
        self.count += 1
        to_add = Node(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index<0:
            index = 0
        elif index>self.count:  return

        if index<self.count-index:
            pred = self.dummy_head
            for _ in range(index):
                pred = pred.next
            succ = pred.next
        else:
            succ = self.dummy_tail
            for _ in range(self.count-index):
                succ = succ.prev
            pred = succ.prev

        self.count += 1
        to_add = Node(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add


    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index<0 or index>=self.count: return
        if index<self.count-index:
            pred = self.dummy_head
            for _ in range(index):
                pred = pred.next
            succ = pred.next.next
        else:
            succ = self.dummy_tail
            for _ in range(self.count-index-1):   #注意range范围
                succ = succ.prev
            pred = succ.prev.prev					#指向删除的元素的前一项
        self.count -= 1
        pred.next = succ
        succ.prev = pred
# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

### 3.链表反转

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while(cur!=None):
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
```

### 4.两两交换链表中的节点

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:           #时间复杂度O(n):对每个节点都要更新，空间复杂度O
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy_head = ListNode(0)
        dummy_head.next = head
        cur = dummy_head
        while cur.next and cur.next.next:
            temp1 = cur.next   
            temp2 = cur.next.next.next  #记录1和3，交换3之前的 1和2

            cur.next = cur.next. 
            cur.next.next = temp1
            cur.next.next.next = temp2
            cur = cur.next.next

        return dummy_head.next  #返回虚拟头节点后面的链表
```

### 5.删除链表的倒数第N个节点

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy_head = ListNode(0)
        dummy_head.next = head
        fast = slow = dummy_head
        for _ in range(n):
            fast = fast.next 
        while fast.next!=None:        
            ##fast先走n步，还差N-n步到最后个链表节点，slow走N-n步，到达删除点的前一个点
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy_head.next
```

### 6.链表相交  

给定两个（单向）链表，判定它们是否相交并返回交点。请注意相交的定义基于节点的引用，而不是基于节点的值。换句话说，如果一个链表的第k个节点与另一个链表的第j个节点是同一节点（引用完全相同），则这两个链表相交。

```PYTHON
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        curA = headA
        curB = headB
        while curA!=curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA
        return curB
```

### 7.环形链表2

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast==slow:
                index2 = fast
                index1 = head
                while index1!=index2:
                    index1 = index1.next
                    index2 = index2.next
                return index1 #index2
        else:   return None
```

### 总结篇

- 链表的存储方式：链表的节点在内存中是分散存储的，通过指针连在一起。.
- 虚拟头节点
- 双指针

# 三.哈希表

### 基础篇

一般哈希表都是用来快速判断一个元素是否出现集合里；

枚举的话时间复杂度是O(n)，但如果使用哈希表的话， 只需要O(1) 就可以做到。

哈希碰撞：hashcode(将名字转为数值)的个数大于哈希表的大小，多个数值对应哈希表的一个索引

一般哈希碰撞有两种解决方法， 拉链法和线性探测法。

拉链法：将发生冲突的元素用链表存储

线性探测法：要保证tablesize大于datasize，在冲突的位置寻找下一个空位放置冲突元素的信息。

哈希表常见的三种数据结构：数组、set(集合)、map(映射)

![image-20210619112719605](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210619112719605.png)

总结一下，**当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法**。

但是哈希法也是**牺牲了空间换取了时间**，因为我们要使用额外的数组，set或者是map来存放数据，才能实现快速的查找。

如果在做面试题目的时候遇到需要判断一个元素是否出现过的场景也应该第一时间想到哈希法！

### 1.有效的字母异位词

给定两个字符串 *s* 和 *t* ，编写一个函数来判断 *t* 是否是 *s* 的字母异位词。

==collections.Counter==:统计字符串中每个字符出现的次数

==defaultdict==函数：dict中若无key，这该key值对应默认值 如defaultdict(int)默认值为0

```python
import operator
class Solution:    #哈希表 数组  时间O(n) 空间O(1)
    def isAnagram(self, s: str, t: str) -> bool:  ##zero和record要分开定义
        record  = [0]*26
        zero = [0]*26
        for i in range(len(s)):
            pos = ord(s[i]) - ord('a')
            record[pos] += 1
        for i in range(len(t)):
            pos = ord(t[i]) - ord('a')
            record[pos] -= 1
        # print(operator.eq(record,zero))
        return True if record==zero else False 
```

```python
class Solution:  #调用函数
    def isAnagram(self, s: str, t: str) -> bool:
        #print(collections.Counter(s))
        #Counter({'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1})
        return collections.Counter(s)==collections.Counter(t)
    	#return sorted(s)==sorted(t)
```

```python
class Solution:  #defaultdict字典
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import defaultdict

        s_dict = defaultdict(int)
        t_dict = defaultdict(int)
        for ch in s:    
            s_dict[ch] += 1
        for ch in t:
            t_dict[ch] += 1
# print(s_dict)   defaultdict(<class 'int'>, {'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1})
        return s_dict==t_dict
```

1.1 赎金信

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

**字典的比较**，比较一个字典的内容是否包含在另一个字典里

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        Note = collections.Counter(ransomNote)
        Mag = collections.Counter(magazine)
        for key in Note:
            if (key not in Mag) or (Note[key]>Mag[key]):
                return False
        return True 
```

1.2 字母异位词分组

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        dic = {}
        for str in strs:
            key = "".join(sorted(str))  
            if key not in dic:
                dic[key] = [str]
            else:
                dic[key].append(str)  #此时字典对应key已有值，用append
        return list(dic.values())
```

1.3 找到字符串中所有字母异位词

给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：字母异位词指字母相同，但排列不同的字符串。不考虑答案输出的顺序。

```python
class Solution:   
    def findAnagrams(self, s: str, p: str) -> List[int]:
        n = len(p)
        p = sorted(p)
        res = []
        for i in range(len(s)-n+1):
            str = s[i:i+n]
            if sorted(str) == p:   
                res.append(i)
        return res
```

```python
class Solution:  #哈希表  时间复杂度 O(n) 空间O(1)
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m,n = len(s),len(p)
        res = []
        s_count = [0]*26
        p_count = [0]*26
        if m<n: return res
        for i in range(n):
            p_count[ord(p[i])-ord('a')] += 1
            s_count[ord(s[i])-ord('a')] += 1
        if s_count==p_count:    res.append(0)
        for i in range(n,m):
            s_count[ord(s[i-n])-ord('a')] -= 1
            s_count[ord(s[i])-ord('a')] += 1
            if s_count == p_count:  res.append(i-n+1)
        return res                
```



### 2.两个数组的交集

给定两个数组，编写一个函数来计算它们的交集。

返回的数组中不包含重复的元素。

注意set()与list()类型不同还需要转化；

set.add()，remove()   &、^、| 取两set中的元素交、并、不同时存在的元素

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
         result_set = set()
         set1 = set(nums1)
         for num in nums2:
             if num in set1:
                 result_set.add(num) 
        return  list(result_set)
        #return list(set(nums1)&set(nums2))
```

2.1 两个数组的交集 II

返回的数组中包含重复的元素。

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        inter = set(nums1) & set(nums2)
        l = []
        for num in inter:
            l += [num] * min(nums1.count(num),nums2.count(num))
        return l
```

### 3.快乐数

编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果 可以变为  1，那么这个数就是快乐数。
如果 n 是快乐数就返回 true ；不是，则返回 false 。

```python
class Solution:  #哈希表 set()
    def isHappy(self, n: int) -> bool:
        sum_set = set()
        while 1:   #一直运行直到return
            res = self.get_sum(n)
            if res == 1:
                return  True
            if res in sum_set:
                return False
            else:
                sum_set.add(res)
            n = res


    def get_sum(self,n):
        res = 0
        while n>0:
            res += (n%10)*(n%10)
            n //= 10
        return res
```

### 4.两数之和 

定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

==hashmap.get(num)== 

```python
class Solution:     #哈希表 map()
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap={}
        for ind,num in enumerate(nums):
            hashmap[num] = ind
        for i,num in enumerate(nums):
            j = hashmap.get(target - num)
            if j is not None and i!=j:
                return [i,j]
```

### 5.四数相加2

给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -228 到 228 - 1 之间，最终结果不会超过 231 - 1 。

```python
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        # use a dict to store the elements in nums1 and nums2 and their sum
        hashmap = dict()
        for n1 in nums1:
            for n2 in nums2:
                if n1 + n2 in hashmap:
                    hashmap[n1+n2] += 1
                else:
                    hashmap[n1+n2] = 1
        
        # if the -(a+b) exists in nums3 and nums4, we shall add the count
        count = 0
        for n3 in nums3:
            for n4 in nums4:
                key = - n3 - n4
                if key in hashmap:
                    count += hashmap[key]
        return count
```

### 6.赎金信

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

==collections.Counter()==

```python
import collections
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        Note = collections.Counter(ransomNote)
        Mag = collections.Counter(magazine)
        for key in Note:
            if Mag[key]<Note[key]: 
                return False
        return True
```

### 7.三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组

注意==sorted(nums)、 nums.sort()==   

一层for循环加双指针，将时间复杂度从暴力解法O(n3)降到O(n2)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)   #nums.sort()
        res = []
        for i in range(len(nums)):           
            left = i + 1
            right = len(nums) - 1
            while left<right:
                target = nums[i]+nums[left]+nums[right]
                if target>0:
                    right -= 1
                elif target<0:
                    left += 1
                else:
                    ok = [nums[i],nums[left],nums[right]]
                    if ok not in res:   res.append(ok)
                    left += 1
                    right -= 1
        return res
```

### 8.四数之和

给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：答案中不可以包含重复的四元组。

两层for循环加双指针，将时间复杂度从暴力解法O(n4)降到O(n3)

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums)
        nums = sorted(nums)
        res = []
        for i in range(n):
            for j in range(i+1,n):
                left = j+1
                right = n-1
                while left<right:
                    Sum = nums[i]+nums[j]+nums[left]+nums[right]
                    if Sum>target:  right -= 1
                    elif Sum<target:    left += 1
                    else:
                        ok = [nums[i],nums[j],nums[left],nums[right]]
                        if ok not in res:   res.append(ok)
                        left += 1
                        right -= 1
        return res
```

### 总结篇

**一般来说哈希表都是用来快速判断一个元素是否出现集合里**。

对于哈希表，要知道**哈希函数**和**哈希碰撞**在哈希表中的作用.

哈希函数是把传入的key映射到符号表的索引上。

哈希碰撞处理有多个key映射到相同索引上时的情景，处理碰撞的普遍方式是拉链法和线性探测法。

接下来是常见的三种哈希结构：

- 数组
- set（集合）
- map（映射）

注意判断什么时候用数组，什么时候用集合以及什么时候用映射



# 四.字符串

### 1.反转字符串

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        n = len(s)
        l,r = 0,n-1
        while l<=r:
            temp = s[l]
            s[l] = s[r]
            s[r] = temp
            ## s[l],s[r] = s[r],s[l]
            l += 1
            r -= 1
```

### 2.反转字符串2

给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:   
        n = len(s)
        i = 0
        s = list(s)
        while i+2*k<n:
            s[i:i+k] = self.reverse(s[i:i+k])
            i += 2*k
        if i+k<n:
            s[i:i+k] = self.reverse(s[i:i+k])
        else:
            s[i:n] = self.reverse(s[i:n])
        return "".join(s)
           
    def reverse(self,s):
        i,j = 0,len(s)-1
        while i<=j:
            # s[i],s[j] = s[j],s[i]
            temp = s[i]
            s[i] = s[j]
            s[j] = temp
            i += 1
            j -= 1
        return s
```

### 3.替换空格

实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

==string.replace( )==

```python
class Solution:  #时间O(n),空间O(n)
    def replaceSpace(self, s: str) -> str:
        # return s.replace(" ","%20")
        res = []
        for ch in s:
            if ch==" ":  res.append('%20')
            else:   res.append(ch)
        return "".join(res)
```

### 4.翻转字符串里的单词

==string.reverse()==反转单词列表

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        l = r = len(s)-1
        res = []
        while l>=0:
            while l>=0 and s[l]!=' ':
                l -= 1
            res.append(s[l+1:r+1])
            while s[l]==' ':    l -= 1
            r = l
        return " ".join(res)   

class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip() # 删除首尾空格
        strs = s.split() # 分割字符串
        strs.reverse() # 翻转单词列表
        return ' '.join(strs) # 拼接为字符串并返回
```

### 5.左旋转字符串

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        s = list(s)
        k = n
        s = self.reversewords(s)
        s[-k:len(s)] = self.reversewords(s[-k:len(s)])
        s[0:(len(s)-k)] = self.reversewords(s[0:(len(s)-k)])
        return "".join(s)

    def reversewords(self,s):
        l,r = 0,len(s)-1
        while l<=r:
            s[l],s[r] = s[r],s[l]
            l += 1
            r -= 1
        return s
        #s = list(s)
        #s[0:n] = reversed(s[0:n])
        #s[n:len(s)] = reversed(s[n:len(s)])
        #print(s)
        #s.reverse()
        #return "".join(s)
```

### 6.实现strStr()  KMP

在一个串中查找是否出现过另一个串，这是KMP的看家本领。

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

**前（后）缀：不包含尾（首）字母的所有子串**  

```python
##注意next数组 前缀表的含义
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a=len(needle)
        b=len(haystack)
        if a==0:
            return 0
        next=self.getnext(a,needle)
        p=-1
        for j in range(b):
            while p>=0 and needle[p+1]!=haystack[j]:
                p=next[p]
            if needle[p+1]==haystack[j]:
                p+=1
            if p==a-1:
                return j-a+1
        return -1

    def getnext(self,a,needle):
        next=['' for i in range(a)]
        k=-1
        next[0]=k
        for i in range(1,len(needle)):
            while (k>-1 and needle[k+1]!=needle[i]):
                k=next[k]
            if needle[k+1]==needle[i]:
                k+=1
            next[i]=k
        return next
```

### 7.重复的子字符串

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s)==0:   return False
        next = self.getnext(s)
        # print(next)
        n = len(s)
        if next[-1]!=-1 and n%(n-next[-1]-1)==0:
            return True
        else:   return False


    def getnext(self,s):
        n = len(s)
        next = ['' for _ in range(n)]
        k = -1
        next[0] = k
        for i in range(1,n):
            while (k>=0 and s[i] != s[k+1]):
                k = next[k]
            if s[i] == s[k+1]:
                k += 1
            next[i] = k
        return next
   
```

### 总结篇



# 五.栈与队列

### 栈与队列基础

栈：先进后出

队列：先进先出

### 1.用栈实现队列 queue

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = list()
        self.stack2 = list() 


    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.stack1.append(x)



    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2.pop()


    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2[-1]


    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.stack1 == [] and self.stack2 == []

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

### 2.用队列实现栈 stack

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（`push`、`top`、`pop` 和 `empty`）。

```python
from collections import deque      #一个队列实现  时间O(1) 空间O(n)
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1 = deque()
        self.queue2 = deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue1.append(x)


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue1.pop()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue1[-1]



    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.queue1) == 0



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

### 3.有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

```PYTHON
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for ch in s:
            if ch == '(':   stack.append(')')
            if ch == '{':   stack.append('}')
            if ch == '[':   stack.append(']')
            if ch == ')':
                if len(stack)!= 0 and stack[-1]== ')':
                    stack.pop()
                else:   return False
            if ch == '}':
                if len(stack)!= 0 and stack[-1]== '}':
                    stack.pop()
                else:   return False
            if ch == ']':
                if len(stack)!= 0 and stack[-1]== ']':
                    stack.pop()
                else:   return False
            else:   continue
        return stack == []
    
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []  # 保存还未匹配的左括号
        mapping = {")": "(", "]": "[", "}": "{"}
        for i in s:
            if i in "([{":  # 当前是左括号，则入栈
                stack.append(i)
            elif stack and stack[-1] == mapping[i]:  # 当前是配对的右括号则出栈
                stack.pop()
            else:  # 不是匹配的右括号或者没有左括号与之匹配，则返回false
                return False
        return stack == []  # 最后必须正好把左括号匹配完
```

### 4.删除字符串中的所有相邻重复项

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for ch in s:
            if stack==[]:   stack.append(ch)
            elif stack[-1] != ch:   stack.append(ch)
            else:   stack.pop()
        return "".join(stack)
```

### 5.逆波兰表达式

已知两数a,b和运算符token：计算值 ==eval('a + token + b')==

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token not in ["+","-","*","/"]:
                stack.append(token)
            else:
                a = stack.pop()
                b = stack.pop()
                res = eval(b + token + a)  #eval('3*3')=9 里面是字符串
                print(res)
                stack.append(str(int(res)))   #注意这里应该要转string
        # print('stack',stack)
        return int(stack[-1])
```

### 6.滑动窗口最大值（二刷）

![image-20210724134148231](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210724134148231.png)

```python
from collections import deque
class myqueue:
    def __init__(self):
        self.queue = deque()
    
    def pop(self,value):
        if self.queue and value==self.queue[0]:
            self.queue.popleft()
    
    def push(self,value):
        while self.queue and value>self.queue[-1]:
            self.queue.pop()
        self.queue.append(value)

    def front(self):
        return self.queue[0]
    

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = myqueue()
        res = []
        for i in range(k):
            queue.push(nums[i])
        res.append(queue.front())
        for i in range(len(nums)-k):
            queue.pop(nums[i])
            queue.push(nums[i+k])
            res.append(queue.front())
        return res
```

### 7.前k个高频元素

对dictionary按value排序，且取前k个key

```python
from collections import defaultdict
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = defaultdict(int)
        res = []
        for num in nums:
            counter[num] += 1
        # print(counter)
        L = sorted(counter.items(),key = lambda i:i[1],reverse = True)  
        #此时的L为一个元组[(1, 3), (2, 2), (3, 1)]
        # print(L)
        for l in L:     
            res.append(l[0])
        return res[:k]
    
```

**小顶堆**

```python
#时间复杂度：O(nlogk)     #空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        map = {}
        for i in range(len(nums)):
            map[nums[i]] = map.get(nums[i],0) + 1 #nums[i]:对应出现的次数
        pri_que = []   #小顶堆
        #用固定大小为k的小顶堆，扫面所有频率的数值
        for key,freq in map.items():
            heapq.heappush(pri_que,(freq,key))
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        # print(pri_que)
        res = [0]*k
        for i in range(k-1,-1,-1):
            res[i] = heapq.heappop(pri_que)[1]
            #找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        return res
```

### 栈与队列总结

滑动窗口题中引入：**单调队列**

栈与队列中引入：**优先级队列**：**就是一个披着队列外衣的堆**

什么是堆呢？

堆是一颗完全二叉树，树中每个结点的值都不小于（或不大于）其左右孩子的值。 如果父亲结点是大于等于左右孩子就是大顶堆，小于等于左右孩子就是小顶堆。

所以大家经常说的大顶堆（堆头是最大元素），小顶堆（堆头是最小元素），如果懒得自己实现的话，就直接用priority_queue（优先级队列）就可以了，底层实现都是一样的，从小到大排就是小顶堆，从大到小排就是大顶堆。



# 六.二叉树

<img src="https://camo.githubusercontent.com/05f375896b965b6c1b2ead25c838b5b3385d18a112878d8e9d3dabacaf2cce8f/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303231393139303830393435312e706e67" alt="二叉树大纲" style="zoom: 50%;" />

### 1.基础篇

种类：**满二叉树和完全二叉树**

优先级队列其实是一个堆，堆就是一棵完全二叉树，同时保证父子节点的顺序关系。

**二叉搜索树**

前面介绍的树，都没有数值的，而二叉搜索树是有数值的了，二叉搜索树是一个有序树。

若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
它的左、右子树也分别为二叉排序树

**平衡二叉搜索树**：又被称为AVL（Adelson-Velsky and Landis）树

且具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

完美二叉树：一个深度为k(>=-1)且有2^(k+1) - 1个结点的二叉树称为完美二叉树

完全二叉树：完全二叉树从根结点到倒数第二层满足完美二叉树，最后一层可以不完全填充，其叶子结点都靠左对齐。

**二叉树的存储方式**

**二叉树可以链式存储 （指针），也可以顺序存储（数组）。**

 **二叉树的遍历方式**

1. 深度优先遍历：先往深走，遇到叶子节点再往回走。 **递归遍历（栈）**
2. 广度优先遍历：一层一层的去遍历。   **迭代遍历（队列）**

![image-20210724194843230](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210724194843230.png)

介绍了二叉树的种类、存储方式、遍历方式以及定义，比较全面的介绍了二叉树各个方面的重点

### 2.二叉树的递归遍历

2.1 二叉树的前序遍历(递归)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

### 前序
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def traverse(root):
            if not root:    return
            else:
                res.append(root.val)
                if root.left:   traverse(root.left)
                if root.right:  traverse(root.right)
        traverse(root)
        return res
### 后序
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def traversal(root):
            if root == None:    return
            # res.append(root.left)
            traversal(root.left)
            traversal(root.right)
            res.append(root.val)
        traversal(root)
        return res
### 中序
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def traversal(root):
            if not root:    return 
            traversal(root.left)
            res.append(root.val)
            traversal(root.right)
        traversal(root)
        return res
```

### 3.二叉树的迭代遍历

**stack = [root]**

```python
###前序   中左右
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        stack = [root]
        if not root:    return []
        res = []
        # print(stack)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
###后序   左右中   == 前序（中左右） 中右左的反序
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        stack = [root]
        if not root:    return []
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
###中序
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = []  # 不能提前将root结点加入stack中
        result = []
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树结点
            if cur:     
                stack.append(cur)
                cur = cur.left		
            # 到达最左结点后处理栈顶结点    
            else:		
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右结点
                cur = cur.right	 #可为None
        return result
```

### 4.二叉树的同一迭代遍历

在（父节点处）加标记

```python
### 前序 
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        st = [root]
        if not root:    return []
        while st:
            node = st.pop()
            if node:
                if node.right:
                    st.append(node.right)  #右
                if node.left:
                    st.append(node.left)  #左
                st.append(node)  #中
                st.append(None)
            else:
                node = st.pop()
                res.append(node.val)
        return res
  ### 后序 左右中  （入栈：中右左）
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        st = [root]
        if not root:    return []
        while st:
            node = st.pop()
            if node:
                st.append(node)
                st.append(None)
                if node.right:
                    st.append(node.right)
                if node.left:
                    st.append(node.left)
            else:
                node = st.pop()
                res.append(node.val)
        return res
    #中序
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #添加右节点（空节点不入栈）
                    st.append(node.right)
                
                st.append(node) #添加中节点
                st.append(None) #中节点访问过，但是还没有处理，加入空节点做为标记。
                
                if node.left: #添加左节点（空节点不入栈）
                    st.append(node.left)
            else: #只有遇到空节点的时候，才将下一个节点放进结果集
                node = st.pop() #重新取出栈中元素
                result.append(node.val) #加入到结果集
        return result
```

### 5.二叉树的层序遍历

102.二叉树的层序遍历

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

**队列先进先出，符合一层一层遍历的逻辑，而是用栈先进后出适合模拟深度优先遍历也就是递归的逻辑。**

```python
class Solution:  #时间 空间都为O(n)
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:    return []
        queue = [root]
        res = []
        while queue:
            out = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                out.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(out)
        #res.reverse()
        #return res[::-1]
        return res
```

107.二叉树的层序遍历2

给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

同102，将res反转即可res.reverse()   return res[::-1]

199.二叉树的右视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

```python
from collections import deque
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:    return []
        queue = deque([root])
        res = []
        while queue:
            node = queue[-1]
            res.append(node.val)
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res
```

637.二叉树的层平均值

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

```python
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root:    return []
        queue = deque([root])
        res = []
        while queue:
            total = 0
            # node = queue.popleft()
            n = len(queue)
            for _ in range(len(queue)):
                node = queue.popleft()
                total += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(total/n)
        return res
```



429.N叉树的层序遍历

给定一个 N 叉树，返回其节点值的*层序遍历*。（即从左到右，逐层遍历）。

树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。

==list.extend(seq)== : extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

```python
from collections import deque
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        res = []
        if not root:    return []
        queue = deque([root])
        while queue:
            # node = queue.popleft()
            out = []
            for _ in range(len(queue)):
                node = queue.popleft()
                out.append(node.val)
                if node.children:
                    queue.extend(node.children)
            res.append(out)
        return res
```

515.在每个树行中找最大值

==map(function,iterable)==  python3 返回迭代器，若作输出需加list()等

```python
from collections import deque
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:    return []
        res = []
        queue = deque([root])
        while queue:
            out = []
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                out.append(node.val)
            # res.append(max(out))  
            res.append(out)
        res = list(map(lambda x:max(x),res))        
        return res
```

116.填充每个节点的下一个右侧节点指针

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:    return[]
        queue = [root]
        while queue:
            n = len(queue)
            for i in range(len(queue)):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == n-1:
                    break
                node.next = queue[0]
        return root
```

117.填充每个节点的下一个右侧节点指针2

### 6.反转二叉树

226.反转二叉树

```python
class Solution:  #层次遍历，交换左右孩子
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:    return None
        queue = [root]
        while queue:
            for _ in range(len(queue)):
                node = queue.pop(0)
                node.left,node.right = node.right,node.left  #交换左右孩子
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
class Solution: ##深度遍历，前序
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:    return None
        stack = [root]
        while stack:
            node = stack.pop()
            node.left ,node.right = node.right,node.left
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return root
class Solution:   ## 递归，前序
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:    return None
        root.left,root.right = root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

### 8.对称二叉树

```python
### 递归
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:    return True
        return self.compare(root.left,root.right)

    def compare(self,left,right):
        if left==None and right!=None:    return False
        elif left!=None and right==None:  return False
        elif left==None and right==None:  return True
        elif left.val!=right.val: return False
        else:
            one = self.compare(left.left,right.right)
            two = self.compare(left.right,right.left)
            return one and two

import collections
class Solution:    ###迭代
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:    return True
        queue = collections.deque()
        queue.append(root.left)
        queue.append(root.right)
        while queue:
            left = queue.popleft()
            right = queue.popleft()
            if not left and not right:
                continue
            if not left or not right or left.val!=right.val:
                return False
            queue.append(left.left)
            queue.append(right.right)
            queue.append(left.right)
            queue.append(right.left)
        return True
```

### 9.二叉树的最大深度

104.二叉树的最大深度

```python
class Solution:  ###递归
    def maxDepth(self, root: TreeNode) -> int:
        if not root:    return 0
        leftnode = self.maxDepth(root.left) 
        rightnode = self.maxDepth(root.right) 
        depth = max(leftnode,rightnode) + 1
        return depth
    
from collections import deque
class Solution:   ###迭代
    def maxDepth(self, root: TreeNode) -> int:
        if not root:    return 0
        queue = deque([root])
        depth = 0
        while queue:
            # node = queue.popleft()
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
        return depth
```

559.N叉树的最大深度

```python
from collections import deque
class Solution:   ###迭代
    def maxDepth(self, root: 'Node') -> int:
        if not root:    return 0
        queue = deque([root])
        depth = 0
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.children:
                    queue.extend(node.children)
        return depth
class Solution:   ###递归
    def maxDepth(self, root: 'Node') -> int:
        if not root:    return 0
        depth = 0
        for i in range(len(root.children)):
            depth = max(depth,self.maxDepth(root.children[i]))
        return depth+1
```

### 10.二叉树的最小深度

```python
from collections import deque
class Solution:   #迭代
    def minDepth(self, root: TreeNode) -> int:
        if not root:    return 0
        queue = deque([root])
        depth = 0
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if not node.left and not node.right:
                    return depth
                    quit()
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)                   
class Solution:   #递归
    def minDepth(self, root: TreeNode) -> int:
        if not root:    return 0
        if not root.left and not root.right:
            return 1
        elif not root.left:
            return self.minDepth(root.right)+1
        elif not root.right:
            return self.minDepth(root.left) +1
        else:
            return min(self.minDepth(root.left),self.minDepth(root.right))+1
```

### 11.完全二叉树的节点个数

```python
from collections import deque
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:    return 0
        queue = deque([root])
        cnt = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                cnt += 1
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
        return cnt
```

### 12.平衡二叉树

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:    return True
        return self.height(root) != -1

    def height(self,node):
        if not node:    return 0
        left_h = self.height(node.left)
        right_h = self.height(node.right)
        if left_h == -1:    return -1
        if right_h == -1:   return -1
        if abs(left_h - right_h)>1: return -1
        else:   return max(left_h,right_h)+1
```

 

### 13.二叉树的所有路径

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:    return []
        paths = []
        def findpath(root,path):
            path += str(root.val)
            if not root.left and not root.right:
                paths.append(path)
            else:
                path += '->'
                if root.left:
                    findpath(root.left,path)
                if root.right:
                    findpath(root.right,path)
        findpath(root,'')
        return paths
```

### 15.相同的树

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        return self.Same(p,q)

    def Same(self,left,right):
        if not left and not right:  return True
        if not left or not right or left.val != right.val:
            return False
        Same_left = self.Same(left.left,right.left)
        Same_right = self.Same(left.right,right.right)
        return Same_left and Same_right
```

### 16.左叶子之和

```python
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        res = 0
        if not root:    return 0
        leftsum = self.sumOfLeftLeaves(root.left)
        rightsum = self.sumOfLeftLeaves(root.right)
        curnodeleft = 0
        if root.left and not root.left.left and not root.left.right:
            curnodeleft = root.left.val
        return curnodeleft + leftsum + rightsum
```

### 17.找树左下角的值

```python
from collections import deque
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:    return None
        queue = deque([root])
        res = 0
        while queue:
            firstleft = queue[0]
            res = firstleft.val        #树的左视图的做法
            for i in range(len(queue)):
                node = queue.popleft()
                # if i == 0:            #最后一行的第一个值的做法
                #     res = node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res
```

### 17.二叉树的右视图

```python
from collections import deque
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:    return []
        queue = deque([root])
        res = []
        while queue:
            node = queue[-1]
            res.append(node.val)
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res
```

### 18.路径总和

112

```python
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:    return False
        def func(root,targetSum):
            if root and not root.left and not root.right and targetSum == root.val:
                return True
            if not root and targetSum != 0: pass
            if root.left:
                if func(root.left,targetSum - root.val):    return True    
                # print('left:hasPathsum({0}，{1})'.format(root.left.val,targetSum-root.val))
            if root.right:
                if func(root.right, targetSum - root.val): return True
                print('right:hasPathsum({0}，{1})'.format(root.right.val,targetSum-root.val))
            return False
        return func(root,targetSum)
```

113.路径总和Ⅱ

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:

        def isornot(root,targetSum):
            if not root.left and not root.right and targetSum == 0:
                res.append(path[:])                 #非 res.append(path) 
            if not root.left and not root.right:
                pass
            if root.left:
                path.append(root.left.val)
                targetSum -= root.left.val
                isornot(root.left,targetSum)
                path.pop()
                targetSum += root.left.val

            if root.right:
                path.append(root.right.val)
                targetSum -= root.right.val
                isornot(root.right,targetSum)
                targetSum += root.right.val
                path.pop()
    
        if not root:    return []
        res,path = [],[]
        path.append(root.val)
        isornot(root,targetSum-root.val)
        return res
```

### 19.构建二叉树

106.从中序与后序遍历序列构造二叉树

![image-20210808152224168](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210808152224168.png)

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 第一步: 特殊情况讨论: 树为空. 或者说是递归终止条件
        if not postorder:   return None   
        # 第二步: 后序遍历的第一个就是当前的中间节点. 
        root_val = postorder[-1]
        root = TreeNode(root_val)
        # 第三步: 找切割点. 
        ind = inorder.index(root_val)
        # 第四步: 切割inorder数组. 得到inorder数组的左,右半边. 
        left_inorder = inorder[:ind] 
        right_inorder = inorder[ind+1:]
        # 第五步: 切割postorder数组. 得到postorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟后序数组大小是相同的. 
        left_postorder =  postorder[:len(left_inorder)]
        right_postorder = postorder[len(left_inorder): -1] #len(postorder)-1]
        assert(len(right_inorder) == len(right_postorder))
        # 第六步: 递归
        root.left = self.buildTree(left_inorder,left_postorder)
        root.right = self.buildTree(right_inorder,right_postorder)

        return root
```

105. 从前序与中序遍历序列构造二叉树

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:    return
        root_val = preorder[0]
        root = TreeNode(root_val)
        ind = inorder.index(root_val)
        left_inorder = inorder[:ind]
        right_inorder = inorder[ind+1:]
        left_preorder = preorder[1:len(left_inorder)+1]
        right_preorder = preorder[len(left_inorder)+1:]
        assert(len(right_preorder)==len(right_inorder))
        root.left = self.buildTree(left_preorder,left_inorder)
        root.right = self.buildTree(right_preorder,right_inorder)
        return root
```

**前序和后序不能唯一确定一颗二叉树！**，因为没有中序遍历无法确定左右部分，也就是无法分割。

tree1 的前序遍历是[1 2 3]， 后序遍历是[3 2 1]。

tree2 的前序遍历是[1 2 3]， 后序遍历是[3 2 1]。

1，2，3 的全左节点； 全右节点   ；前后序均相同，但是树不同

### 20.最大二叉树

```python
class Treenode:
    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if len(nums)==0:    return
        root_val = max(nums)
        root = TreeNode(root_val)
        ind = nums.index(root_val)
        left_nums = nums[:ind]
        right_nums = nums[ind+1:]
        root.left = self.constructMaximumBinaryTree(left_nums)
        root.right = self.constructMaximumBinaryTree(right_nums)
        return root
```









### 30.删除二叉搜索树中的节点(真题)

```python
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return root  
        if root.val == key:  
            if not root.left and not root.right: 
                root = None
                return root
            if not root.left and root.right:  
                root = root.right
                return root
            if root.left and not root.right:  
                root = root.left
                return root
            if root.left and root.right:
                v = root.left
                while v.right:
                    v = v.right
                v.right = root.right
                root = root.left
                return root
        root.left = self.deleteNode(root.left,key)  
        root.right = self.deleteNode(root.right,key)  
        return root
```



# 七.回溯算法

<img src="C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210808155735685.png" alt="image-20210808155735685" style="zoom:50%;" />



### 1.基础篇

回溯法也可以叫做回溯搜索法，它是一种搜索的方式。

回溯是递归的副产品，只要有递归就会有回溯。

所以以下讲解中，回溯函数也就是递归函数，指的都是一个函数。

**回溯的本质是穷举，穷举所有可能，然后选出我们想要的答案**  效率不高

回溯法，一般可以解决如下几种问题：

- 组合问题：N个数里面按一定规则找出k个数的集合
- 切割问题：一个字符串按一定规则有几种切割方式
- 子集问题：一个N个数的集合里有多少符合条件的子集
- 排列问题：N个数按一定规则全排列，有几种排列方式
- 棋盘问题：N皇后，解数独等等

### 2.组合

### 3.组合优化

```python
class Solution:   #回溯  时间 O(c(n,k)*k) 空间O(n)
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(n,k,ind):
            if len(path[:])==k:
                paths.append(path[:])
                return None
            #for i in range(ind,n+1):
            for i in range(ind,n-(k-len(path))+2):  #组合优化 剪枝
                path.append(i)
                backtrack(n,k,i+1)
                path.pop()  
        paths = []
        path = []
        backtrack(n,k,1)
        return paths
```

### 4.电话号码的字母组合

注意得使用全局变量self.s ； lettermap的使用

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        self.s = ""
        lettermap = ["","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
        if len(digits)==0:  return res
        def traceback(digits,index):
            if index == len(digits):
                res.append(self.s)
                return res
            digit = int(digits[index])
            letters = lettermap[digit]
            for letter in letters:
                self.s += letter
                traceback(digits,index+1)
                self.s = self.s[:-1]     #回溯
        
        traceback(digits,0)
        return res
```

###  6.组合总和

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        def traceback(candidates,target,sum,startindex):
            if sum>target:  return 
            if sum==target: res.append(path[:])
            for i in range(startindex,len(candidates)):
                if sum + candidates[i] > target:    return
                sum += candidates[i]
                path.append(candidates[i])
                traceback(candidates,target,sum,i)
                sum -= candidates[i]
                path.pop()
            
        candidates = sorted(candidates)
        traceback(candidates,target,0,0)
        return res
```

### 7.组合总和2

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        paths = []
        path = []
        numsum = 0
        def getpath(candidates,target,numsum,index):
            if numsum>target:
                return
            if numsum == target:
                paths.append(path[:])
            for i in range(index,len(candidates)):
                if numsum + candidate[i]>target:	return  #剪枝
                if i>index and candidates[i]==candidates[i-1]:
                    continue
                path.append(candidates[i])
                numsum += candidates[i]
                getpath(candidates,target,numsum,i+1)
                path.pop()
                numsum -= candidates[i]
            return 
        getpath(candidates,target,0,0)
        return paths
```

### 9.分割回文串

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        n = len(s)
        def partsubstring(s,startindex):
            if startindex >= len(s):
                res.append(path[:])
            for i in range(startindex,n):
                substr = s[startindex:i+1]
                if substr == substr[::-1]:
                    path.append(substr)
                else:   continue
                partsubstring(s,i+1)
                path.pop()

        partsubstring(s,0)
        return res
```

### 10.复原IP地址

```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if len(s)>12 or len(s)<4:
            return []
        ans = []
        path = []
        def backtrack(path, startIndex):
            if len(path) == 4:
                if startIndex == len(s):
                    ans.append(".".join(path[:]))
                    return
            for i in range(startIndex+1, min(startIndex+4, len(s)+1)):  # 剪枝
                string = s[startIndex:i]
                if not 0 <= int(string) <= 255:
                    continue
                if not string == "0" and not string.lstrip('0') == string:
                    continue
                path.append(string)
                backtrack(path, i)
                path.pop()

        backtrack([], 0)
        return ans
```

### 11.子集

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def traceback(path,startindex):
            res.append(path[:])
            for i in range(startindex,len(nums)):
                path.append(nums[i])
                traceback(path,i+1)
                path.pop()
        
        traceback(path,0)
        return res
```

### 13.子集2

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        def traceback(path,startindex):
            #if sorted(path) not in res:
            #    res.append(sorted(path[:]))
            res.append(path[:])
            for i in range(startindex,len(nums)):
				if i > startindex and nums[i]==nums[i-1]:
                    continue
                path.append(nums[i])
                traceback(path,i+1)
                path.pop()
        nums = sorted(nums)
        traceback([],0)
        return res
```

### 14.递增子序列

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def traceback(nums,startindex):
            repeat = []
            if len(path)>1:
                res.append(path[:])
            for i in range(startindex,len(nums)):
                if nums[i] in repeat:
                    continue
                if len(path)>=1:
                    if nums[i]<path[-1]:
                        continue
                path.append(nums[i])
                repeat.append(nums[i])
                traceback(nums,i+1)
                path.pop()
        
        traceback(nums,0)
        return res
```

### 15.全排列

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def traceback(nums):
            if len(path) == len(nums):
                res.append(path[:])
            for i in range(0,len(nums)):
                if nums[i] in path:
                    continue
                path.append(nums[i])
                traceback(nums)
                path.pop()

        traceback(nums)
        return res
```

### 16.全排列2

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # res用来存放结果
        # if not nums: return []
        res = []
        used = [0] * len(nums)
        def backtracking(nums, used, path):
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if not used[i]:
                    if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                        continue
                    used[i] = 1
                    path.append(nums[i])
                    backtracking(nums, used, path)
                    path.pop()
                    used[i] = 0
        # 记得给nums排序
        backtracking(sorted(nums),used,[])
        return res
```



### 19.重新安排行程

### 20.N皇后

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n:   return []
        board = [['.']*n for _ in range(n)]
        res = []
        def isValid(board,row,col):  #判断该位置的这一行是否可行
            for i in range(len(board)):  #这一列
                if board[i][col] == 'Q':
                    return False
            #左上角
            i = row - 1
            j = col - 1
            while i>=0 and j>=0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            #左下角
            i = row - 1
            j = col + 1
            while i>=0 and j<len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def traceback(board,row,n):
            if row == n:
                path = []
                for temp in board:
                    temp_str = "".join(temp)
                    path.append(temp_str)
                res.append(path)
            for col in range(n):
                if not isValid(board,row,col):
                    continue
                board[row][col] = 'Q'
                traceback(board,row+1,n)
                board[row][col] = '.'
        
        traceback(board,0,n)
        return res
```

### 21.解数独

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def isValid(board,row,col,num):
            # row 判断行
            for i in range(9):
                if board[row][i] == str(num):
                    return False
            # col
            for i in range(9):
                if board[i][col] == str(num):
                    return False
            # 3X3
            startrow = (row // 3)*3
            startcol = (col // 3)*3
            for r in range(startrow,startrow+3):
                for c in range(startcol,startcol+3):
                    if board[r][c] == str(num):
                        return False
            return True
        
        def traceback(board):
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] != '.':  continue
                    for k in range(1,10):
                        if isValid(board,i,j,k):
                            board[i][j] = str(k)
                            if traceback(board):
                                return True
                            board[i][j] = '.'
                    return False
            return True

        traceback(board)
```



## 八.贪心算法

### 1.基础篇

### 2.分发饼干

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        s = sorted(s)
        g = sorted(g)
        count = 0
        start = len(s) - 1
        for index in range(len(g)-1,-1,-1):
            if start>=0 and s[start]>=g[index]:   #优先将大饼干分给大胃口
                start -= 1
                count += 1
        return count
```

### 3.摆动序列

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。

给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        res = 1 #默认最右端有一个峰值
        preC,curC = 0,0
        for i in range(1,len(nums)):
            curC = nums[i] - nums[i-1]  #当前的符号
            if preC*curC <= 0 and curC != 0:
                res += 1 
                preC = curC
        return res
```

### 4.最大子序和

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```python
class Solution:   #时间O(n)  空间O(1)
    def maxSubArray(self, nums: List[int]) -> int:
        res = nums[0]
        count = 0
        for index in range(len(nums)):
            count += nums[index]
            res = max(res,count)
            if count < 0:   count = 0  #贪心算法，若前子数组和小于0，则丢掉前子数组
        return res
```

### 5.小结

贪心的本质是选择每一阶段的局部最优，从而达到全局最优。（无反例）

### 6.买卖股票的最佳时机Ⅱ

给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

思路：将利润分解为以每天为单位

```python
class Solution:  #时间O(n),空间O(1)
    def maxProfit(self, prices: List[int]) -> int:
        result = 0
        for i in range(1, len(prices)):
            result += max(prices[i] - prices[i - 1], 0)  #只收集正利润的区间，即为最终利润
        return result
```

### 7.跳跃游戏

给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums)==1:    return True
        i = 0
        cover = 0
        while i <= cover:   # python不支持动态修改for循环中变量,使用while循环代替
            cover = max(i + nums[i],cover)
            if cover >= len(nums)-1:    return True
            i += 1
        return False
```

### 8.跳跃游戏Ⅱ

给你一个非负整数数组 nums ，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums)==1:    return 0
        cur_cover,next_cover = 0,0   #当前能走的最远距离和下一步能走的最长距离
        res = 0   #步数
        for i in range(len(nums)):
            next_cover = max(i+nums[i],next_cover)  #统计下一步的最长距离
            if i == cur_cover:  #如果到了这一步的最远的覆盖区域，则走一步
                res += 1
                if next_cover >= len(nums)-1:   break  #到达数组最后个位置
                cur_cover = next_cover
        return res
```



### 9.K次取反后最大化的数组和

给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引 i 并将 A[i] 替换为 -A[i]，然后总共重复这个过程 K 次。（我们可以多次选择同一个索引 i。）

以这种方式修改数组后，返回数组可能的最大和。

==sorted(nums,key=abs)==按绝对值大小给数组排序

```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        nums = sorted(nums,key=abs)    
        count = 0
        for i in range(len(nums)-1,-1,-1):
            if count < k:
                if nums[i] < 0:
                    nums[i] = -nums[i]
                    count += 1
        while count < k:
            nums[0] = -nums[0]
            count += 1
        return sum(nums)
```

### 小结

### 11.加油站

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

```python
class Solution:   #时间O(n) 空间O(n)
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        rem = []
        for i in range(len(gas)):
            rem.append(gas[i]-cost[i])
        if sum(rem) < 0:    return -1
        start,curSum = 0,0
        for i in range(len(rem)):   #判断数组从start开始 相加，和为正
            curSum += rem[i]
            if curSum < 0:
                curSum = 0
                start = i + 1
        return start         
```

### 12.分发糖果

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candyVec = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candyVec[i] = candyVec[i - 1] + 1
        for j in range(len(ratings) - 2, -1, -1):
            if ratings[j] > ratings[j + 1]:
                candyVec[j] = max(candyVec[j], candyVec[j + 1] + 1)
        return sum(candyVec)
```

### 13.柠檬水找零

在柠檬水摊上，每一杯柠檬水的售价为 5 美元。顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。注意，一开始你手头没有任何零钱。

给你一个整数数组 bills ，其中 bills[i] 是第 i 位顾客付的账。如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        changes = [0 for _ in range(2)]
        for bill in bills:
            if bill == 5:
                changes[0] += 1
            if bill == 10:
                changes[0] -= 1
                changes[1] += 1
            if bill == 20:
                if changes[1]>0:
                    changes[1] -= 1
                    changes[0] -= 1
                elif changes[1] == 0:
                    changes[0] -= 3
            for change in changes:
                if change<0:   return False
        return True
```





## 九.动态规划

### 1.基础

**对于动态规划问题，我将拆解为如下五步曲，这五步都搞清楚了，才能说把动态规划真的掌握了！**

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

### 2.斐波那契数

```python
class Solution:
    def fib(self, n: int) -> int:
        if n<2: return n
        Fib = [0]*(n+1)
        Fib[1] = 1
        for i in range(2,n+1):
            Fib[i] = Fib[i-1] + Fib[i-2]
        return Fib[n]
```

### 3.爬楼梯

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n<=2:    return n
        dp = [0]*(n+1)
        dp[1],dp[2] = 1,2
        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

### 4.使用最小花费爬楼梯

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0]*(n+1)
        for i in range(2,n+1):
            dp[i] = min(cost[i-1]+dp[i-1],cost[i-2]+dp[i-2])
        return dp[n]
```

### 6.不同路径

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0] = [1 for _ in range(n)]
        for i in range(m):  dp[i][0] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        # print(dp)
        return dp[-1][-1]
```

### 7.不同路径Ⅱ

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m,n = len(obstacleGrid),len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:   break
        for i in range(n):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:   break
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i][j-1] + dp[i-1][j]
        return dp[-1][-1]
```

### 8.整数拆分

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

```python
class Solution:   #大于等于4的数 可以分为2和3（优先）的数字的和，已达到乘积最大
    def integerBreak(self, n: int) -> int:
        if n<2: return 0
        if n==2:    return 1
        if n==3:    return 2
        y = n//3
        n = n - 3*y
        if y>0 and n==1:
            y -= 1
            n += 3
        x = n//2
        return pow(2,x)*pow(3,y)
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[2] = 1
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            # 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
        for i in range(3,n+1):
            for j in range(1,i):
                dp[i] = max(dp[i],j*dp[i-j],j*(i-j))
        return dp[n]
```



































# 剑值offer

### 15.二进制中1的个数

一个数 减去 1后，如（1100  -》 1011）  二进制表示中最后一位的1变为0，最后一位的1后面的0变为1，再取一个 位与运算&，（1100&1011 -》 1000），二进制中的1就少了一个，能执行多少次这样的操作就有多少个1

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = (n-1)&n
            count += 1
        return count
```

### 面试题16.05 阶乘尾数

设计一个算法，算出 n 阶乘有多少个尾随零。

n的阶乘结果有多少个0，取决于有多少个5（阶乘结果中2的个数肯定比5多，2*5产生0）

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n:
            count += n//5
            n = n//5    #要注意25=5*5，有两个因子5
        return count
```

### 48.最长不含重复字符的子字符串 （蔚来真题）







# 刷题中的笔记



## 函数

**不引入变量交换两值：**

```python
#python 
##众所皆知，Python中的变量并不直接存储值，而只是引用一个内存地址，交换变量时，只是交换了引用的地址 ，所以支持下面这种方式：
y, x = x, y
print(x, y)
#数学知识
x = x + y   
y = x - y   #等价于x+y-y
x = x - y   #此处的y存储的已经是原来的x的引用了，所以等价于x+y-(x)
#位运算
x = x ^ y
y = x ^ y   #等价于x^y^y 结果是x
x = x ^ y   #此处的y存储的已经是原来的x的引用了，所以等价于x^y^x 结果是y
```

用列表实现栈和队列：

==from collections import deque==

```python
 		# deque来自collections模块，不在力扣平台时，需要手动写入
        # 'from collections import deque' 导入
        # deque相比list的好处是，list的pop(0)是O(n)复杂度，deque的popleft()是O(1)复杂度
#列表实现栈  先进先出
stack = [1,2,3]
stack.append(4)  #进栈
stack.pop() #出栈
#列表实现队列
from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")
queue.popleft()
```

==sort\sorted==

sorted()不改变原数据， .sort()改变原数据

底层实现为归并排序，时间复杂度O(Nlog2N)

==dict.get()==

![image-20210724183735582](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210724183735582.png)

==.items()==

items() 方法的遍历：items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回。

==list.extend(seq)== :

extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

==map(function,iterable)==  python3 返回迭代器，若作输出需加list()等

**map()** 会根据提供的函数对指定序列做映射。

第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

==list and list[:]==

list“赋值”时会用到list2 = list1 或者 list2[:] = list1，前者两个名字指向同一个对象，后者两个名字指向不同对象

```python
##生成24位小写字母
#List
[chr(i) for i in range(97,123)]
#String
''.join([chr(i) for i in range(97,123)])
```



## 方法

递归步骤

- 确定递归函数的参数和返回值

- 确定终止条件

- 确定单层递归的逻辑

回溯

- 回溯函数模板返回值以及参数

- 回溯函数终止条件
- 回溯搜索的遍历过程

## 倒序















# 牛客网刷题

## 机考处理键盘输入输出

1. 处理一行键盘输入

**num = list(map(int,input().split()))**

**nums = input("")    num = [int(n) for n in nums.split()]** 

```python
#对于多元输入
n,k,m = map(int,input().split())
#法一
line = list(map(str,input().split())) #将输入转化成列表，以空格为分隔符

#L=[] 
#L.append(map(int,input().split()))#将输入存入列表

#读入一维矩阵
arr = input("")    #输入一个一维数组，每个数之间使空格隔开
num = [int(n) for n in arr.split()]    #将输入每个数以空格键隔开做成数组
print(num)        #打印数组

import sys
try:
    while True:
        line = sys.stdin.readline().strip()
        if line == '':
            break
        lines = line.split()
        print int(lines[0]) + int(lines[1])
except:
    pass

import sys 
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
```

2.处理多行（矩阵）键盘输入

```python
#读入二维矩阵  适用于n*n矩阵 
## 已知输入行数
n=int(input())
m=[]
for i in range(n):
    m.append(list(map(int,input().split())))  
    #m.append(list(map(float,input().split(" "))))
    
n = int(input())        #输入二维数组的行数和列数
m = [[0]*n]*n        #初始化二维数组
for i in range(n):
    m[i] = input().split(" ")       #输入二维数组，同行数字用空格分隔，不同行则用回车换行
    #m[i] = list(map(str,input().split(" ")))  输入二维数组，同行数字用空格分隔，不同行则用回车换行

#未知行数
import sys
try:
    mx = []
    while True:
        # m = input().strip()
        m = sys.stdin.readline().strip()
        #若是多输入，strip()默认是去除首尾空格，返回一个包含多个字符串的list。
        if m == '':
            break
        m = list(m.split())
        mx.append(m)
    print(mx)
except:
    pass
```

3.处理字符串的输入

str = input().split(" ")		print(" ".join(str))

 join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。 str.join(sequence)



怎么读这样的输入：

![image-20210718164147118](C:\Users\Ivan\AppData\Roaming\Typora\typora-user-images\image-20210718164147118.png)

4.输出保留前几位小数

```python
#方法1：
print("%.2f" % 0.13333)

#方法2  保留两位有效数字
print("{:.2}".format(0.13333))    

#方法3
round(0.13333, 2)

```





## 题目

1.老师想知道从某某同学当中，分数最高的是多少，现在请你编程模拟老师的询问。当然，老师有时候需要更新某位同学的成绩.

输入包括多组测试数据。每组输入第一行是两个正整数N和M（0 < N <= 30000,0 < M < 5000）,分别代表学生的数目和操作的数目。学生ID编号从1编到N。第二行包含N个整数，代表这N个学生的初始成绩，其中第i个数代表ID为i的学生的成绩。接下来又M行，每一行有一个字符C（只取‘Q’或‘U’），和两个正整数A,B,当C为'Q'的时候, 表示这是一条询问操作，他询问ID从A到B（包括A,B）的学生当中，成绩最高的是多少。当C为‘U’的时候，表示这是一条更新操作，要求把ID为A的学生的成绩更改为B。

对于每一次询问操作，在一行里面输出最高成绩.

```python
while True:
    try:
        a, b = map(int, input().split())
        grades = list(map(int, input().split()))
        for i in range(b):
            command = input().split()
            if command[0] == "Q":
                start, end = sorted([int(command[1]), int(command[2])])
                print(max(grades[start - 1:end]))
            else: grades[int(command[1]) - 1] = int(command[2])
    except:
        break
```

2.反转链表

输入一个链表，反转链表后，输出新链表的表头。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        pEnd,p1 = None,None
        while pHead is not None:
            p1 = pHead.next
            pHead.next = pEnd
            pEnd = pHead
            pHead = p1
        return pEnd   
```

3.斐波那契数列

4.二分查找

```python
class Solution:
    def search(self , nums , target ):
        # write code here
        n = len(nums)
        left,right = 0,n-1
        while right>left:
            mid = (right+left)//2
            if nums[mid]==target:
                while mid>0 and nums[mid]==nums[mid-1]:
                    mid -= 1
                return mid
            if nums[mid]>target:
                left = mid + 1
            if nums[mid]<target:
                right = mid - 1
        return -1  
```

5.排序

https://www.runoob.com/w3cnote/ten-sorting-algorithm.html

```python
## 冒泡法  超时
class Solution:
    def MySort(self , arr ):
        # write code here
        for n in range(1,len(arr)):
            for i in range(0,len(arr)-n):
                if arr[i]>arr[i+1]:
                    arr[i],arr[i+1] = arr[i+1],arr[i]   #注意这种交换两值的方式
        return arr
## 选择排序法
class Solution:
    def MySort(self , arr ):
        # write code here 
        index,n = 0,len(arr)
        for i in range(n):
            small = arr[i]
            for j in range(i,n):
                if arr[j]<=small:
                    small = arr[j]
                    index = j
            arr[i],arr[index] = arr[index],arr[i]
        return arr
```

6.寻找第K大

有一个整数数组，请你根据快速排序的思路，找出数组中第K大的数。

给定一个整数数组a,同时给定它的大小n和要找的K(K在1到n之间)，请返回第K大的数，保证答案存在。

```python
class Solution:
    def findKth(self, a, n, K):
        # write code here
        for i in range(K):
            index = i
            for j in range(i,n):   
                if a[j]>a[index]:
                    index = j
            a[i],a[index] = a[index],a[i]
        print(a)
        return a[K-1]
```



## 真题

### 华为实习

1.压缩字符串系列

1.1字符串压缩

字符串压缩。利用字符重复出现的次数，编写一种方法，实现基本的字符串压缩功能。比如，字符串aabcccccaaa会变为a2b1c5a3。若“压缩”后的字符串没有变短，则返回原先的字符串。你可以假设字符串中只包含大小写英文字母（a至z）。

```python
class Solution:
    def compressString(self, S: str) -> str:
        CS = ""
        i = 0
        while i<len(S):
            count = 1
            while i<len(S)-1 and S[i]==S[i+1]:
                count +=1
                i = i+1
            seq = [S[i],str(count)]
            zishu = ''.join(seq)
            CS = CS + zishu
            i= i+1
        if len(CS)>=len(S):  return S
        else:   return CS        
```

1.2压缩字符串

给定一组字符，使用原地算法将其压缩。

压缩后的长度必须始终小于或等于原数组长度。

数组的每个元素应该是长度为1 的字符（不是 int 整数类型）。

在完成原地修改输入数组后，返回数组的新长度。

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        n = len(chars)
        i,count = 0,1
        for j in range(1,n+1):      #i取到最后个字符时，不用再判断是否有下一个数字与当前数字相同
            if  j<n and chars[j] == chars[j-1]:
                count += 1
            else:
                chars[i] = chars[j-1]
                i += 1
                if count>1:
                    for num in str(count):
                        chars[i] = num
                        i += 1
                count = 1
        return i
```

1.3字符串的排列

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        c, res = list(s), []
        def dfs(x):
            if x == len(c) - 1:
                res.append(''.join(c))   # 添加排列方案
                return
            dic = set()
            for i in range(x, len(c)):
                if c[i] in dic: continue # 重复，因此剪枝
                dic.add(c[i])
                c[i], c[x] = c[x], c[i]  # 交换，将 c[i] 固定在第 x 位
                dfs(x + 1)               # 开启固定第 x + 1 位字符
                c[i], c[x] = c[x], c[i]  # 恢复交换
        dfs(0)
        return res
```













2.二叉树分枝

3.最大路径和





### 美团

1.判断数组能否构成1到n的排列

2.在字符串尾端添加最少字符个数  形成回文串

leetcode1312 让字符串成为回文串的最少插入次数

给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 s 成为回文串的 最少操作次数 。

「回文串」是正读和反读都相同的字符串。

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for span in range(2,n+1): #span<2 res=0
            for i in range(n-span+1):
                j = span + i -1  #j-i+1=span
                dp[i][j] = min(dp[i+1][j],dp[i][j-1])+1
                #dp[i][j] = dp[i+1][j] + 1   chuanwei
                if s[i]==s[j]:
                    dp[i][j] = min(dp[i][j],dp[i+1][j-1])
        return dp[0][n-1]
```





3.机器人整时刻碰撞爆炸消失；输入m行机器人状态[10,R]... 输出爆炸时间

4.打车最少时间；  在n个节点

选择题：	

高斯混合模型      

生成对抗网络  seq2seq  pix2pix cyclegan  GCNN

SGD:  能否跳过局部最小值点？ 收敛速度与batchsize成正比？  收敛速度慢，振幅大？



### momenta

1）层次遍历反转二叉树

2）从[1,n]产生m个不重复的随机整数  哈希表



### 蔚来

1）最大不重复子字符串长度  剑指offer48

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:   return 0
        num = [0]*len(s)
        def getlength(s):
            count = 0
            repeat = set()
            for ch in s:
                if ch in repeat:
                    break
                count += 1
                repeat.add(ch)
            return count

        for i in range(len(s)):
            num[i] = getlength(s[i:])
        return max(num)
```



2）删除二叉树中特定值的节点

```python
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return root  
        if root.val == key:  
            if not root.left and not root.right: 
                root = None
                return root
            if not root.left and root.right:  
                root = root.right
                return root
            if root.left and not root.right:  
                root = root.left
                return root
            if root.left and root.right:
                v = root.left
                while v.right:
                    v = v.right
                v.right = root.right
                root = root.left
                return root
        root.left = self.deleteNode(root.left,key)  
        root.right = self.deleteNode(root.right,key)  
        return root
```



3）有序链表合并



### 华为

1.输入包缓存包，最终输出保留的包的个数

3.矩阵中求联通0所需的最小代价

### 小米

230题

### 商汤

算法题：使用高斯消元法求解线性方程组
给定一个线性方程组AX=B，其中A为nxn的方阵，B为nx1的列向量，使用高斯消元法求其根X。
高斯消元法：对增广矩阵[A,B]进行行变换（交换两行，给某一行乘以非0系数，将某一行乘以系数加在另一行上），最终变为[A',B']的形式，其中A'为上三角矩阵。再反解A'X=B'即可。
说明：若方程有唯一解，返回nx1的列向量，为方程的根。若方程无解或有多个解，返回False

算法题：多边形的面积
给定二维平面内的连续k个坐标(x_k,y_k)，求他们围成的多边形的面积。
算法题：连续正整数的和
给定一个正整数N，试求有多少组连续的正整数，满足所有的数字之和为N？
示例：
输入：5
输出：2
解释：[5] [2,3]
参考：最优时间复杂度o(n^(1/2)) 空间复杂度o(1)
注：给出的坐标按照顺序1-2-3-4-...-1连线一定可以围成多边形，多边形可能是凸的，也可能是凹的。



# 面试经历

## momenta

1.coding面

车道线检测；

设计IOU， gt+pred像素数值相加，  IOU设计为 2的个数除以1的个数； 

lr与batch size的大小关系

如何评估车道线

row-wise如何检测水平线，  水平拓宽 与 膨胀

coding： 翻转二叉树

2.算法面

pytorch里的op操作

一个类里如果没有先定义self.conv，能否直接调用

coding： n个数字里面生成m个不同的整数（哈希表）

