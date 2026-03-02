# 图/DFS/BFS 题型

## 一句话结论
DFS/BFS 是图遍历的两种核心范式，DFS 用递归栈/显式栈实现路径探索，BFS 用队列实现层级遍历，岛屿数量是经典连通分量问题，可用 DFS、BFS 或并查集三种方式求解，时间复杂度均为 O(m×n)。

---

## 核心定义/公式

### 图遍历复杂度
- **DFS/BFS 时间复杂度**：O(V + E)，V 为节点数，E 为边数
- **二维网格复杂度**：O(m × n)，m 行 n 列
- **空间复杂度**：
  - DFS：递归栈深度 O(V) 最坏，网格 O(min(m, n)) 平均
  - BFS：队列长度 O(V) 最坏，网格 O(m × n) 最坏
  - 并查集：O(m × n) parent 数组 + 路径压缩后接近 O(α(mn))

### 连通分量计数
```
连通分量数 = 遍历启动次数
每次遍历标记一个完整连通区域，启动次数即为答案
```

### 并查集核心操作
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # 初始连通分量数
    
    def find(self, x):
        # 路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # 按秩合并
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1
```

---

## 为什么（3 个因果链）

### 1. 为什么岛屿数量是连通分量问题？
- **现象**：网格中相邻的 '1' 构成岛屿
- **根因**：连通分量的定义是"互相可达的节点集合"，相邻 '1' 可达即属于同一分量
- **结果**：统计连通分量数 = 统计岛屿数，可用图遍历算法求解

### 2. 为什么 DFS/BFS/并查集都能解？
- **现象**：三种方法结果相同
- **根因**：
  - DFS：深度优先探索，递归/栈保证访问完一个连通块
  - BFS：广度优先，队列逐层扩展访问完连通块
  - 并查集：动态维护连通关系，合并后统计独立集合数
- **结果**：都是"标记已访问 + 扩展邻居"的不同实现，本质等价

### 3. 为什么实际面试常选 DFS？
- **现象**：岛屿数量 DFS 解法代码最简洁
- **根因**：递归天然适合"探索邻居 → 继续探索"的模式，无需显式维护队列
- **结果**：DFS 代码量最少，但需注意递归栈溢出风险，大网格可能需要 BFS 或显式栈 DFS

---

## 怎么做（可落地步骤）

### 标准做法：DFS 递归

**步骤**：
1. 遍历网格每个位置
2. 遇到未访问的 '1'，岛屿数 +1，启动 DFS 标记整个连通块
3. DFS 中将当前位置改为 '0'（或标记 visited），递归访问上下左右邻居
4. 继续遍历，直到所有位置处理完毕

**代码模板**：
```python
def numIslands(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        # 边界检查 + 水域检查
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
            return
        # 标记已访问（原地修改）
        grid[i][j] = '0'
        # 四个方向
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    
    return count
```

**关键配置/参数**：
- 方向数组：`directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]`（四连通）
- 八连通：增加 `[(1, 1), (1, -1), (-1, 1), (-1, -1)]`
- 是否允许原地修改：允许则改 '1' 为 '0'，不允许则用 visited 数组

---

### 标准做法：BFS

**步骤**：
1. 遍历网格，遇到 '1' 启动 BFS
2. BFS 用队列存储待访问位置
3. 出队时标记并扩展邻居，邻居是 '1' 则入队
4. 队列为空时完成一个连通块

**代码模板**：
```python
from collections import deque

def numIslands_bfs(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                # BFS
                queue = deque([(i, j)])
                grid[i][j] = '0'  # 标记已访问
                
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                            grid[nx][ny] = '0'
                            queue.append((nx, ny))
    
    return count
```

---

### 标准做法：并查集

**步骤**：
1. 初始化并查集，每个 '1' 独立为一个集合
2. 遍历网格，对每个 '1' 合并其右侧和下方的 '1'
3. 最终集合数即为岛屿数

**代码模板**：
```python
class UnionFind:
    def __init__(self, grid):
        m, n = len(grid), len(grid[0])
        self.parent = []
        self.rank = [0] * (m * n)
        self.count = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.parent.append(i * n + j)
                    self.count += 1
                else:
                    self.parent.append(-1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1


def numIslands_uf(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    uf = UnionFind(grid)
    directions = [(1, 0), (0, 1)]  # 只需向右和向下
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                idx = i * n + j
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                        nidx = ni * n + nj
                        uf.union(idx, nidx)
    
    return uf.count
```

---

### DFS 显式栈版本（防栈溢出）

```python
def numIslands_stack(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                stack = [(i, j)]
                grid[i][j] = '0'
                
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                            grid[nx][ny] = '0'
                            stack.append((nx, ny))
    
    return count
```

---

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **DFS 递归** | 代码最简洁，易于理解和手写 | 递归栈可能溢出（大网格），最坏空间 O(m×n) | 小中型网格（< 10⁶ 节点），面试手撕首选 |
| **BFS** | 队列显式管理，无栈溢出风险，适合求最短路径 | 代码稍长，最坏空间 O(m×n) | 大网格，或需要层级遍历/最短路径时 |
| **DFS 显式栈** | 无栈溢出风险，保持 DFS 逻辑 | 代码比递归 DFS 长 | 大网格 + 需要保持 DFS 特性 |
| **并查集** | 适合动态连通性查询，可处理多次 union/find | 代码最长，常用于需要动态维护连通关系的场景 | 多次连通查询、动态合并、离线查询 |

**选择建议**：
- 面试手撕：优先 DFS 递归（代码最短）
- 生产环境大网格：BFS 或显式栈 DFS
- 需要多次查询连通性：并查集

---

## 高频追问（8 个）

### 1. Q: DFS 和 BFS 的区别是什么？
A: 
- **遍历顺序**：DFS 深度优先（一条路走到黑），BFS 广度优先（逐层扩展）
- **数据结构**：DFS 用栈（递归栈或显式栈），BFS 用队列
- **应用场景**：DFS 适合路径搜索、连通性问题；BFS 适合最短路径、层级遍历
- **空间复杂度**：DFS 空间与递归深度相关，BFS 空间与最大层宽相关

### 2. Q: 什么时候用 DFS，什么时候用 BFS？
A:
- **DFS**：连通性判断、路径搜索（所有路径）、拓扑排序、回溯类问题
- **BFS**：最短路径（无权图）、层级遍历、最小步数问题
- **并查集**：动态连通性、多次合并/查询、Kruskal MST

### 3. Q: 为什么 DFS 递归可能栈溢出？怎么解决？
A:
- **原因**：Python 默认递归深度约 1000，大网格递归深度可能超限
- **解决方案**：
  1. 使用显式栈实现 DFS
  2. 使用 BFS
  3. 增大递归限制 `sys.setrecursionlimit(10**6)`（不推荐，治标不治本）

### 4. Q: 岛屿数量并查集解法的时间复杂度是多少？
A:
- **初始化**：O(m × n)
- **遍历合并**：O(m × n × α(m × n))，α 是反阿克曼函数，实际接近常数
- **总复杂度**：O(m × n) 实际，理论 O(m × n × α(m × n))
- 路径压缩 + 按秩合并后，单次 find/union 接近 O(1)

### 5. Q: 如何处理斜向连接（八连通）的岛屿？
A:
- 修改方向数组：四连通 → 八连通
```python
# 四连通
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# 八连通
directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]
```
- 复杂度不变，但连通块数可能减少

### 6. Q: 岛屿问题的变体有哪些？
A:
1. **岛屿周长**：统计每个 '1' 的边界贡献（相邻 '1' 数量）
2. **岛屿最大面积**：DFS/BFS 时计数，记录最大值
3. **统计封闭岛屿**：先排除边界连通块，再统计内部岛屿
4. **不同岛屿形状计数**：用路径序列/签名表示形状，去重
5. **岛屿网格变形**：允许改变部分 '0' → '1'，求最大岛屿

### 7. Q: 如果网格非常大，无法一次性加载到内存怎么办？
A:
- **分块处理**：将网格分成块，逐块加载
- **并查集 + 边界信息**：维护块边界节点的连通关系
- **外部排序思想**：类似归并，多次扫描合并
- **工程优化**：使用生成器/yield、数据库存储、流式处理

### 8. Q: DFS 访问顺序影响结果吗？
A:
- **连通分量数**：不影响，每个连通块必然被完整标记
- **路径搜索**：影响找到的路径（DFS 找到的不是最短路径）
- **时间/空间**：影响实际访问顺序，但复杂度不变
- **最短路径**：DFS 不保证，必须用 BFS

---

## 常见错误（5 个）

### 1. 边界检查遗漏或错误
**错误**：
```python
# 错误：只检查了 < 边界，没检查 >= 边界
if i < m and j < n and grid[i][j] == '1':
    # ...
```
**正确**：
```python
# 必须检查双向边界
if 0 <= i < m and 0 <= j < n and grid[i][j] == '1':
    # ...
```

### 2. 访问标记时机错误
**错误**：
```python
# 错误：入队时未标记，可能导致重复入队
queue.append((nx, ny))
grid[nx][ny] = '0'  # 标记太晚
```
**正确**：
```python
# 正确：入队时立即标记
grid[nx][ny] = '0'
queue.append((nx, ny))
```

### 3. 方向数组遗漏或重复
**错误**：
```python
# 错误：方向不全或重复
directions = [(1, 0), (0, 1)]  # 只有两个方向
```
**正确**：
```python
# 正确：完整的四连通
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
```

### 4. 并查集 parent 数组初始化错误
**错误**：
```python
# 错误：所有位置都初始化为索引，包括水域
self.parent = list(range(m * n))
self.count = m * n
```
**正确**：
```python
# 正确：只有陆地才计入连通分量
for i in range(m):
    for j in range(n):
        if grid[i][j] == '1':
            self.parent.append(i * n + j)
            self.count += 1
        else:
            self.parent.append(-1)  # 水域标记为 -1
```

### 5. 递归深度未考虑栈溢出
**错误**：
```python
# 错误：大网格直接递归可能栈溢出
def dfs(i, j):
    # ...
    dfs(i+1, j)  # Python 默认递归深度 ~1000
```
**正确**：
- 小网格：递归 DFS
- 大网格：BFS 或显式栈 DFS
- 或提示面试官：`sys.setrecursionlimit(10**6)`（说明风险）

---

## 反问面试官的问题

### 技术深度类
1. "如果网格是稀疏的（大部分是水），有没有更高效的空间优化方案？比如只存储陆地坐标？"
2. "如果需要支持动态添加/删除陆地（在线查询），并查集如何改进？需要支持撤销操作吗？"
3. "对于三维或高维网格，DFS/BFS 的空间复杂度如何变化？有什么优化思路？"

### 业务场景类
1. "实际业务中，这类连通性问题通常出现在什么场景？比如社交网络、图像处理、地图应用？"
2. "如果需要实时统计岛屿数量变化（动态添加陆地），如何设计数据结构？"
3. "在分布式环境下处理超大规模网格图，如何并行化？MapReduce 思路适用吗？"

---

## 自测题

### 口述（能流畅讲清楚）
1. DFS 和 BFS 的核心区别（遍历顺序、数据结构、应用场景）
2. 为什么岛屿数量可以用三种方法求解（本质等价性）
3. 并查集的路径压缩和按秩合并如何优化复杂度
4. 什么时候选择 DFS vs BFS vs 并查集

### 手写（5 分钟能写出）
1. **岛屿数量 DFS 解法**（最常考）
2. **岛屿最大面积**（BFS 计数变体）
3. **岛屿周长计算**（边界贡献统计）
4. **DFS 显式栈版本**（防栈溢出）
5. **并查集基本操作**（find/union 路径压缩）

---

## 标签
#handwrite #leetcode #图 #DFS #BFS #并查集 #连通分量 #阿里

---

## 相关文档
- [[01-回溯题型]] - DFS 是回溯的基础，路径搜索问题
- [[02-滑窗题型]] - BFS 的层级遍历与滑窗的滑动逻辑
- [[03-DP题型]] - 完全背包（零钱兑换）与图的最短路关系
- [[../07-分布式训练ZeRO/01-并行策略总览]] - 图的分布式处理思路

---

## 题目索引（LeetCode）

| 题号 | 题目名 | 难度 | 核心考点 |
|------|--------|------|----------|
| 200 | 岛屿数量 | 中 | DFS/BFS/并查集，连通分量 |
| 695 | 岛屿的最大面积 | 中 | DFS/BFS 计数 |
| 463 | 岛屿周长 | 简 | 边界贡献统计 |
| 1254 | 统计封闭岛屿数目 | 中 | 先排除边界连通块 |
| 694 | 不同岛屿的数量 | 中 | 形状签名/序列化 |
| 733 | 图像渲染 | 简 | BFS/DFS 基础 |
| 130 | 被围绕的区域 | 中 | 边界 DFS/BFS |
| 417 | 太平洋大西洋水流问题 | 中 | 反向 DFS/BFS |
| 127 | 单词接龙 | 难 | BFS 最短路径 |
| 207 | 课程表 | 中 | 拓扑排序/环检测 |

---

## 扩展：图遍历常见变体

### 1. 最短路径问题（无权图）
- **BFS** 天然适合，逐层扩展
- 时间复杂度：O(V + E)
- 应用：单词接龙（LeetCode 127）

### 2. 拓扑排序
- **DFS** 后序遍历逆序
- **Kahn 算法**（BFS + 入度）
- 应用：课程表（LeetCode 207/210）

### 3. 环检测
- **DFS** 三色标记（未访问/访问中/已完成）
- **并查集**：同一连通分量中试图 union
- 应用：课程表（LeetCode 207）

### 4. 多源 BFS
- 多个起点同时开始 BFS
- 应用： Walls and Gates（LeetCode 286）、01 矩阵（LeetCode 542）

### 5. 双向 BFS
- 从起点和终点同时 BFS
- 空间优化，适合搜索空间大的场景
- 应用：单词接龙优化

### 6. DFS + 回溯
- 找所有路径/组合
- 应用：路径搜索、组合生成

---

## 快速代码模板速查

### DFS 递归模板
```python
def dfs(grid, i, j, visited):
    if not (0 <= i < m and 0 <= j < n):
        return
    if visited[i][j] or grid[i][j] == '0':
        return
    visited[i][j] = True
    for dx, dy in directions:
        dfs(grid, i + dx, j + dy, visited)
```

### BFS 模板
```python
from collections import deque

def bfs(grid, i, j):
    queue = deque([(i, j)])
    visited[i][j] = True
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
```

### 并查集模板
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```