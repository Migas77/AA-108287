#### 3 Tasks

- **Programming** - write function
- **Analyze function result**
  - Formal: Summations -> Formula
  - Experimental: Table
  - After the previous two get Complexity Class
- **Efficiency** - count the number of iterations
  - Formal
  - Experimental


```py
int f1(int n) {
  int i,r=0;
  for(i = 1; i <= n; i++)
    r += i;
  return r;
}
```

```py
int f2(int n) {
  int i,j,r=0;
  for(i = 1; i <= n; i++)
    for(j = 1; j <= n; j++)
      r += 1;
  return r;
}
```

```py
int f3(int n) {
  int i,j,r=0;
  for(i = 1; i <= n; i++)
    for(j = i; j <= n; j++)
      r += 1;
  return r;
}
```

```py
int f4(int n) {
  int i,j,r=0;
  for(i = 1; i <= n; i++)
    for(j = 1; j <= i; j++)
      r += j;
  return r;
}
```

#### Result
| n    | f1 | f2 | f3 | f4 |
| ---- | -- | -- | -- | -- |
| 1    | 1  | 1  | 1  | 1  |
| 2    | 3  | 4  | 3  | 4  |
| 3    | 6  | 9  | 6  | 10 |
| 4    | 10 | 16 | 10 | 20 |
| 5    | 15 | 25 | 15 | 35 |
| 6    | 21 | 36 | 21 | 56 |
| 7    | 28 | 49 | 28 | 84 |
| 8    | 36 | 64 | 36 | 120|
| 9    | 45 | 81 | 45 | 165|
| 10   | 55 | 100| 55 | 220|

- **f1** - somatõrio de 1 até n = n(n+1)/2p, O(n^2)
- **f2** - n^2, O(n^2)
- **f3** - O(n^2)
- **f4** - 