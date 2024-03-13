pool: list = [9, 9, 8, 2, 4, 4, 3, 5, 3, 1, 1, 4, 5, 1, 4, 1, 9, 2, 6, 0, 8, 1, 7]
ptr: int = 0


def randint(l: int, r: int) -> int:
  global pool
  global ptr
  seed: int = pool[ptr]
  ptr = (ptr + 1) % len(pool)
  return seed % (r - l) + l


def randdouble(l: float, r: float) -> float:
  global pool
  global ptr
  seed1: int = pool[ptr]
  ptr = (ptr + 1) % len(pool)
  seed2: int = pool[ptr]
  ptr = (ptr + 1) % len(pool)
  if seed1 > seed2:
    return seed2 / seed1 * (r - l - 0.1) + l
  return seed1 / seed2 * (r - l - 0.1) + l
