def two_sum(nums,target):
    seen={}
    for i,n in enumerate(nums):
        if target-n in seen:
            return [seen[target-n],i]
        seen[n]=i
    return None

print(two_sum([2,7,11,15],9))