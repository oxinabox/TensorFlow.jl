
k = placeholder(Float32; shape=[10,20, -1])
@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)

@test_throws ErrorException get_shape(placeholder(Float32), 1)


# Reading nodes
sess = Session(Graph())
X = placeholder(Float32; shape=[10])
W = get_variable("W", [10,20], Float32)
Y=X*W
run(sess, initialize_all_variables())
x_val=rand(Float32, (1,10))
w_val = run(sess, W)
@test size(w_val)==(10,20)
y_val = run(sess, Y, Dict(X=>x_val))
@test y_val ≈ x_val * w_val

# make sure can get with feed 
w_val3  = run(sess, [W], Dict(X=>x_val))
# make sure can get both at once
y_val2, w_val2  = run(sess, [Y, W], Dict(X=>x_val))
@test y_val2 == y_val
@test w_val2 == w_val
