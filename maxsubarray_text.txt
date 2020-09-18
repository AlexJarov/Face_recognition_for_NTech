#Programm finding maxsimal submassive
#by Jarov Alexey Valeryevich
#Algoritm requires one loop across array, so it has complexity of O(n),
#That is better than plain brute force of each submassive,
#wich has complexity O(n^2)

def findMaxSubArray(A): #defining function findMaxSubArray, it requires a one-dimesional array as an argument
#in case no array was suplied, return None
    if len(A)==0:
        print("No numbers was written")
        return None
#if array has just one nuber, it allways maximal, so return that number
    elif len(A)==1:
        print(A)
        return A
#for all other arrays
    else: #defining starting parameters
        partsum = A[0]
        maxsum = A[0]
        counter = 1
        probmaxL = 0
        maxL = 0
        maxR = 1
#starting loop
        for i in range (1,len(A)):
            number = A[i]
            previous = A[i-1]
#if array starts with negative number, to ensure that number won't fall in resulting submassive when it
#shouldn't, we take second number as starting if second number is greater
            if (previous < 0) and (i==1):
                if number < 0:
                    partsum = max(partsum,number)
                else:
                    partsum = 0
            else:
                counter += 1
#Trying to add new number to sum
            partsum += A[i]
#If that sum is greater than previous, it becomes new sum
            if partsum >= maxsum:
                maxL = i - counter + 1
                maxR = i + 1
                maxsum = partsum
#Else we restarting counter
            if partsum < maxsum:
                if partsum < 0:
                    counter = 0
                    partsum = 0
#New step of the loop
            i += 1
#Output of the function
        return A[maxL:maxR]

#Now we inputing array by keyboard
A = list(map(int,input("Enter the numbers of array splitting them by Backspase, press Enter when done").split( )))
#And here is the result:
print(findMaxSubArray(A))
