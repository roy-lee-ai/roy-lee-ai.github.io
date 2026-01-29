---
title: "My Pythonic Code Diary"
date: 2025-01-21
categories: [Python]
tags: [python]     # TAG names should always be lowercase
---

## How to make use of `defaultdict(list)`?

A `defaultdict` is a specialized container found in Python's built-in `collections` module. It works exactly like a standard dictionary, but with one major advantage: **it never raises a `KeyError`.**

{: .prompt-info }
> When you initialize it as `defaultdict(list)`, the `list` constructor acts as a **default_factory**. If a requested key is not found, it automatically creates a new, empty list `[]` for that key.


## Real-World Example: Group Anagrams

This is a classic LeetCode Medium problem (LeetCode 49) often seen in **IBM Research** coding assessments.

```python
from collections import defaultdict

def group_anagrams(strs):
    # Initialize defaultdict with list factory
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Create a unique key by sorting the string
        key = "".join(sorted(s))
        
        # No need to check if key exists; just append!
        anagram_map[key].append(s)
        
    return list(anagram_map.values())

# Input: ["eat","tea","tan","ate","nat","bat"]
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

## Accessing Dictionary Values: `m[key]` vs. `m.get(key)`

The main difference is how they handle **missing keys**.

* **`m[key]` (Direct Access):**
    * Raises a **`KeyError`** if the key is missing.

* **`m.get(key)` (Safe Access):**
    * Returns **`None`** (or a custom default) if the key is missing.

---

## In Python, slicing ([:]) is specifically designed to be "forgiving" and will not raise an IndexError even if the range exceeds the list's length.

```python
# Scenario: 100 images, batch_size = 30
for i in range(0, 100, 30):
    # i will be 0, 30, 60, 90
    batch = data[i : i + 30] 
    
    # During the final iteration (i = 90):
    # data[90 : 120] is called.
    # Python sees the list ends at 99, so it safely 
    # returns the final 10 items (90-99) as a batch.
```



s
