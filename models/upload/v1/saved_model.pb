
Á%Ł%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
;
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.13.12
b'unknown'Ő
o
input_tensorPlaceholder*
shape:˙˙˙˙˙˙˙˙˙/*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙/
i
fc1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"/      
[
fc1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ł=ž
[
fc1/random_uniform/maxConst*
valueB
 *Ł=>*
dtype0*
_output_shapes
: 
Ą
 fc1/random_uniform/RandomUniformRandomUniformfc1/random_uniform/shape*
dtype0*
_output_shapes
:	/*
seed2ýŞÄ*
seedą˙ĺ)*
T0
n
fc1/random_uniform/subSubfc1/random_uniform/maxfc1/random_uniform/min*
T0*
_output_shapes
: 

fc1/random_uniform/mulMul fc1/random_uniform/RandomUniformfc1/random_uniform/sub*
T0*
_output_shapes
:	/
s
fc1/random_uniformAddfc1/random_uniform/mulfc1/random_uniform/min*
_output_shapes
:	/*
T0


fc1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	/*
	container *
shape:	/
­
fc1/kernel/AssignAssign
fc1/kernelfc1/random_uniform*
use_locking(*
T0*
_class
loc:@fc1/kernel*
validate_shape(*
_output_shapes
:	/
p
fc1/kernel/readIdentity
fc1/kernel*
T0*
_class
loc:@fc1/kernel*
_output_shapes
:	/
X
	fc1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
v
fc1/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

fc1/bias/AssignAssignfc1/bias	fc1/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@fc1/bias
f
fc1/bias/readIdentityfc1/bias*
T0*
_class
loc:@fc1/bias*
_output_shapes	
:


fc1/MatMulMatMulinput_tensorfc1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
{
fc1/BiasAddBiasAdd
fc1/MatMulfc1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
X
	bn1/ConstConst*
valueB*  ?*
dtype0*
_output_shapes	
:
w
	bn1/gamma
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

bn1/gamma/AssignAssign	bn1/gamma	bn1/Const*
use_locking(*
T0*
_class
loc:@bn1/gamma*
validate_shape(*
_output_shapes	
:
i
bn1/gamma/readIdentity	bn1/gamma*
T0*
_class
loc:@bn1/gamma*
_output_shapes	
:
Z
bn1/Const_1Const*
valueB*    *
dtype0*
_output_shapes	
:
v
bn1/beta
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

bn1/beta/AssignAssignbn1/betabn1/Const_1*
use_locking(*
T0*
_class
loc:@bn1/beta*
validate_shape(*
_output_shapes	
:
f
bn1/beta/readIdentitybn1/beta*
T0*
_class
loc:@bn1/beta*
_output_shapes	
:
Z
bn1/Const_2Const*
dtype0*
_output_shapes	
:*
valueB*    
}
bn1/moving_mean
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ą
bn1/moving_mean/AssignAssignbn1/moving_meanbn1/Const_2*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
{
bn1/moving_mean/readIdentitybn1/moving_mean*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:
Z
bn1/Const_3Const*
valueB*  ?*
dtype0*
_output_shapes	
:

bn1/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
˝
bn1/moving_variance/AssignAssignbn1/moving_variancebn1/Const_3*
use_locking(*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes	
:

bn1/moving_variance/readIdentitybn1/moving_variance*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes	
:
l
"bn1/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:

bn1/moments/meanMeanfc1/BiasAdd"bn1/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
d
bn1/moments/StopGradientStopGradientbn1/moments/mean*
T0*
_output_shapes
:	

bn1/moments/SquaredDifferenceSquaredDifferencefc1/BiasAddbn1/moments/StopGradient*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
&bn1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
Ş
bn1/moments/varianceMeanbn1/moments/SquaredDifference&bn1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
m
bn1/moments/SqueezeSqueezebn1/moments/mean*
_output_shapes	
:*
squeeze_dims
 *
T0
s
bn1/moments/Squeeze_1Squeezebn1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:
X
bn1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:
j
bn1/batchnorm/addAddbn1/moments/Squeeze_1bn1/batchnorm/add/y*
T0*
_output_shapes	
:
U
bn1/batchnorm/RsqrtRsqrtbn1/batchnorm/add*
T0*
_output_shapes	
:
c
bn1/batchnorm/mulMulbn1/batchnorm/Rsqrtbn1/gamma/read*
_output_shapes	
:*
T0
m
bn1/batchnorm/mul_1Mulfc1/BiasAddbn1/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
bn1/batchnorm/mul_2Mulbn1/moments/Squeezebn1/batchnorm/mul*
T0*
_output_shapes	
:
b
bn1/batchnorm/subSubbn1/beta/readbn1/batchnorm/mul_2*
T0*
_output_shapes	
:
u
bn1/batchnorm/add_1Addbn1/batchnorm/mul_1bn1/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
	bn1/ShapeShapefc1/BiasAdd*
_output_shapes
:*
T0*
out_type0
a
bn1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
bn1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
bn1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bn1/strided_sliceStridedSlice	bn1/Shapebn1/strided_slice/stackbn1/strided_slice/stack_1bn1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
bn1/Rank/packedPackbn1/strided_slice*
T0*

axis *
N*
_output_shapes
:
J
bn1/RankConst*
dtype0*
_output_shapes
: *
value	B :
Q
bn1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
Q
bn1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
f
	bn1/rangeRangebn1/range/startbn1/Rankbn1/range/delta*
_output_shapes
:*

Tidx0
c
bn1/Prod/inputPackbn1/strided_slice*
T0*

axis *
N*
_output_shapes
:
i
bn1/ProdProdbn1/Prod/input	bn1/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Z
bn1/CastCastbn1/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
N
	bn1/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ĺ ?
D
bn1/subSubbn1/Cast	bn1/sub/y*
T0*
_output_shapes
: 
J
bn1/truedivRealDivbn1/Castbn1/sub*
T0*
_output_shapes
: 
X
bn1/mulMulbn1/moments/Squeeze_1bn1/truediv*
T0*
_output_shapes	
:

bn1/AssignMovingAvg/decayConst*
valueB
 *
×#<*"
_class
loc:@bn1/moving_mean*
dtype0*
_output_shapes
: 

)bn1/AssignMovingAvg/bn1/moving_mean/zerosConst*
valueB*    *"
_class
loc:@bn1/moving_mean*
dtype0*
_output_shapes	
:
¨
bn1/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@bn1/moving_mean*
	container *
shape:
Ý
bn1/moving_mean/biased/AssignAssignbn1/moving_mean/biased)bn1/AssignMovingAvg/bn1/moving_mean/zeros*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

bn1/moving_mean/biased/readIdentitybn1/moving_mean/biased*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:

,bn1/moving_mean/local_step/Initializer/zerosConst*"
_class
loc:@bn1/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
˘
bn1/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@bn1/moving_mean*
	container *
shape: 
ă
!bn1/moving_mean/local_step/AssignAssignbn1/moving_mean/local_step,bn1/moving_mean/local_step/Initializer/zeros*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(

bn1/moving_mean/local_step/readIdentitybn1/moving_mean/local_step*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes
: 
Ş
'bn1/AssignMovingAvg/bn1/moving_mean/subSubbn1/moving_mean/biased/readbn1/moments/Squeeze*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:
ź
'bn1/AssignMovingAvg/bn1/moving_mean/mulMul'bn1/AssignMovingAvg/bn1/moving_mean/subbn1/AssignMovingAvg/decay*
_output_shapes	
:*
T0*"
_class
loc:@bn1/moving_mean
Ţ
3bn1/AssignMovingAvg/bn1/moving_mean/bn1/moving_mean	AssignSubbn1/moving_mean/biased'bn1/AssignMovingAvg/bn1/moving_mean/mul*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:*
use_locking( 

3bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd/valueConst*
valueB
 *  ?*"
_class
loc:@bn1/moving_mean*
dtype0*
_output_shapes
: 
ă
-bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd	AssignAddbn1/moving_mean/local_step3bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd/value*
use_locking( *
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes
: 
ü
(bn1/AssignMovingAvg/bn1/moving_mean/readIdentitybn1/moving_mean/biased.^bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd4^bn1/AssignMovingAvg/bn1/moving_mean/bn1/moving_mean*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:
ú
+bn1/AssignMovingAvg/bn1/moving_mean/sub_1/xConst.^bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd4^bn1/AssignMovingAvg/bn1/moving_mean/bn1/moving_mean*
valueB
 *  ?*"
_class
loc:@bn1/moving_mean*
dtype0*
_output_shapes
: 
˝
)bn1/AssignMovingAvg/bn1/moving_mean/sub_1Sub+bn1/AssignMovingAvg/bn1/moving_mean/sub_1/xbn1/AssignMovingAvg/decay*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes
: 
ý
*bn1/AssignMovingAvg/bn1/moving_mean/read_1Identitybn1/moving_mean/local_step.^bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd4^bn1/AssignMovingAvg/bn1/moving_mean/bn1/moving_mean*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes
: 
Ę
'bn1/AssignMovingAvg/bn1/moving_mean/PowPow)bn1/AssignMovingAvg/bn1/moving_mean/sub_1*bn1/AssignMovingAvg/bn1/moving_mean/read_1*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes
: 
ú
+bn1/AssignMovingAvg/bn1/moving_mean/sub_2/xConst.^bn1/AssignMovingAvg/bn1/moving_mean/AssignAdd4^bn1/AssignMovingAvg/bn1/moving_mean/bn1/moving_mean*
dtype0*
_output_shapes
: *
valueB
 *  ?*"
_class
loc:@bn1/moving_mean
Ë
)bn1/AssignMovingAvg/bn1/moving_mean/sub_2Sub+bn1/AssignMovingAvg/bn1/moving_mean/sub_2/x'bn1/AssignMovingAvg/bn1/moving_mean/Pow*
_output_shapes
: *
T0*"
_class
loc:@bn1/moving_mean
Ő
+bn1/AssignMovingAvg/bn1/moving_mean/truedivRealDiv(bn1/AssignMovingAvg/bn1/moving_mean/read)bn1/AssignMovingAvg/bn1/moving_mean/sub_2*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:
˝
)bn1/AssignMovingAvg/bn1/moving_mean/sub_3Subbn1/moving_mean/read+bn1/AssignMovingAvg/bn1/moving_mean/truediv*
T0*"
_class
loc:@bn1/moving_mean*
_output_shapes	
:
š
bn1/AssignMovingAvg	AssignSubbn1/moving_mean)bn1/AssignMovingAvg/bn1/moving_mean/sub_3*
_output_shapes	
:*
use_locking( *
T0*"
_class
loc:@bn1/moving_mean

bn1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<*&
_class
loc:@bn1/moving_variance
Ś
/bn1/AssignMovingAvg_1/bn1/moving_variance/zerosConst*
valueB*    *&
_class
loc:@bn1/moving_variance*
dtype0*
_output_shapes	
:
°
bn1/moving_variance/biased
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *&
_class
loc:@bn1/moving_variance*
	container *
shape:
ď
!bn1/moving_variance/biased/AssignAssignbn1/moving_variance/biased/bn1/AssignMovingAvg_1/bn1/moving_variance/zeros*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(

bn1/moving_variance/biased/readIdentitybn1/moving_variance/biased*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes	
:

0bn1/moving_variance/local_step/Initializer/zerosConst*&
_class
loc:@bn1/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
Ş
bn1/moving_variance/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *&
_class
loc:@bn1/moving_variance*
	container *
shape: 
ó
%bn1/moving_variance/local_step/AssignAssignbn1/moving_variance/local_step0bn1/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes
: 

#bn1/moving_variance/local_step/readIdentitybn1/moving_variance/local_step*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes
: 
Ź
-bn1/AssignMovingAvg_1/bn1/moving_variance/subSubbn1/moving_variance/biased/readbn1/mul*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes	
:
Î
-bn1/AssignMovingAvg_1/bn1/moving_variance/mulMul-bn1/AssignMovingAvg_1/bn1/moving_variance/subbn1/AssignMovingAvg_1/decay*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes	
:
ö
=bn1/AssignMovingAvg_1/bn1/moving_variance/bn1/moving_variance	AssignSubbn1/moving_variance/biased-bn1/AssignMovingAvg_1/bn1/moving_variance/mul*
_output_shapes	
:*
use_locking( *
T0*&
_class
loc:@bn1/moving_variance
Ś
9bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd/valueConst*
valueB
 *  ?*&
_class
loc:@bn1/moving_variance*
dtype0*
_output_shapes
: 
÷
3bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd	AssignAddbn1/moving_variance/local_step9bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*&
_class
loc:@bn1/moving_variance

.bn1/AssignMovingAvg_1/bn1/moving_variance/readIdentitybn1/moving_variance/biased4^bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd>^bn1/AssignMovingAvg_1/bn1/moving_variance/bn1/moving_variance*
_output_shapes	
:*
T0*&
_class
loc:@bn1/moving_variance

1bn1/AssignMovingAvg_1/bn1/moving_variance/sub_1/xConst4^bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd>^bn1/AssignMovingAvg_1/bn1/moving_variance/bn1/moving_variance*
dtype0*
_output_shapes
: *
valueB
 *  ?*&
_class
loc:@bn1/moving_variance
Ď
/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_1Sub1bn1/AssignMovingAvg_1/bn1/moving_variance/sub_1/xbn1/AssignMovingAvg_1/decay*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes
: 

0bn1/AssignMovingAvg_1/bn1/moving_variance/read_1Identitybn1/moving_variance/local_step4^bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd>^bn1/AssignMovingAvg_1/bn1/moving_variance/bn1/moving_variance*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes
: 
ŕ
-bn1/AssignMovingAvg_1/bn1/moving_variance/PowPow/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_10bn1/AssignMovingAvg_1/bn1/moving_variance/read_1*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes
: 

1bn1/AssignMovingAvg_1/bn1/moving_variance/sub_2/xConst4^bn1/AssignMovingAvg_1/bn1/moving_variance/AssignAdd>^bn1/AssignMovingAvg_1/bn1/moving_variance/bn1/moving_variance*
valueB
 *  ?*&
_class
loc:@bn1/moving_variance*
dtype0*
_output_shapes
: 
á
/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_2Sub1bn1/AssignMovingAvg_1/bn1/moving_variance/sub_2/x-bn1/AssignMovingAvg_1/bn1/moving_variance/Pow*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes
: 
ë
1bn1/AssignMovingAvg_1/bn1/moving_variance/truedivRealDiv.bn1/AssignMovingAvg_1/bn1/moving_variance/read/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_2*
T0*&
_class
loc:@bn1/moving_variance*
_output_shapes	
:
Ń
/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_3Subbn1/moving_variance/read1bn1/AssignMovingAvg_1/bn1/moving_variance/truediv*
_output_shapes	
:*
T0*&
_class
loc:@bn1/moving_variance
É
bn1/AssignMovingAvg_1	AssignSubbn1/moving_variance/bn1/AssignMovingAvg_1/bn1/moving_variance/sub_3*
_output_shapes	
:*
use_locking( *
T0*&
_class
loc:@bn1/moving_variance
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
h
bn1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
Q
bn1/cond/switch_tIdentitybn1/cond/Switch:1*
_output_shapes
: *
T0

O
bn1/cond/switch_fIdentitybn1/cond/Switch*
T0
*
_output_shapes
: 
S
bn1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
ą
bn1/cond/Switch_1Switchbn1/batchnorm/add_1bn1/cond/pred_id*
T0*&
_class
loc:@bn1/batchnorm/add_1*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
q
bn1/cond/batchnorm/add/yConst^bn1/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *o:
|
bn1/cond/batchnorm/addAddbn1/cond/batchnorm/add/Switchbn1/cond/batchnorm/add/y*
T0*
_output_shapes	
:
¨
bn1/cond/batchnorm/add/SwitchSwitchbn1/moving_variance/readbn1/cond/pred_id*"
_output_shapes
::*
T0*&
_class
loc:@bn1/moving_variance
_
bn1/cond/batchnorm/RsqrtRsqrtbn1/cond/batchnorm/add*
T0*
_output_shapes	
:
|
bn1/cond/batchnorm/mulMulbn1/cond/batchnorm/Rsqrtbn1/cond/batchnorm/mul/Switch*
T0*
_output_shapes	
:

bn1/cond/batchnorm/mul/SwitchSwitchbn1/gamma/readbn1/cond/pred_id*
T0*
_class
loc:@bn1/gamma*"
_output_shapes
::

bn1/cond/batchnorm/mul_1Mulbn1/cond/batchnorm/mul_1/Switchbn1/cond/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
bn1/cond/batchnorm/mul_1/SwitchSwitchfc1/BiasAddbn1/cond/pred_id*
T0*
_class
loc:@fc1/BiasAdd*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
~
bn1/cond/batchnorm/mul_2Mulbn1/cond/batchnorm/mul_2/Switchbn1/cond/batchnorm/mul*
T0*
_output_shapes	
:
˘
bn1/cond/batchnorm/mul_2/SwitchSwitchbn1/moving_mean/readbn1/cond/pred_id*
T0*"
_class
loc:@bn1/moving_mean*"
_output_shapes
::
|
bn1/cond/batchnorm/subSubbn1/cond/batchnorm/sub/Switchbn1/cond/batchnorm/mul_2*
T0*
_output_shapes	
:

bn1/cond/batchnorm/sub/SwitchSwitchbn1/beta/readbn1/cond/pred_id*
T0*
_class
loc:@bn1/beta*"
_output_shapes
::

bn1/cond/batchnorm/add_1Addbn1/cond/batchnorm/mul_1bn1/cond/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

bn1/cond/MergeMergebn1/cond/batchnorm/add_1bn1/cond/Switch_1:1*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
U

relu1/ReluRelubn1/cond/Merge*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
fc2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
[
fc2/random_uniform/minConst*
valueB
 *qÄž*
dtype0*
_output_shapes
: 
[
fc2/random_uniform/maxConst*
valueB
 *qÄ>*
dtype0*
_output_shapes
: 
Ą
 fc2/random_uniform/RandomUniformRandomUniformfc2/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
*
seed2ŢP*
seedą˙ĺ)
n
fc2/random_uniform/subSubfc2/random_uniform/maxfc2/random_uniform/min*
T0*
_output_shapes
: 

fc2/random_uniform/mulMul fc2/random_uniform/RandomUniformfc2/random_uniform/sub*
T0* 
_output_shapes
:

t
fc2/random_uniformAddfc2/random_uniform/mulfc2/random_uniform/min*
T0* 
_output_shapes
:



fc2/kernel
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
*
	container *
shape:

Ž
fc2/kernel/AssignAssign
fc2/kernelfc2/random_uniform*
T0*
_class
loc:@fc2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
q
fc2/kernel/readIdentity
fc2/kernel* 
_output_shapes
:
*
T0*
_class
loc:@fc2/kernel
X
	fc2/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
v
fc2/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

fc2/bias/AssignAssignfc2/bias	fc2/Const*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes	
:
f
fc2/bias/readIdentityfc2/bias*
T0*
_class
loc:@fc2/bias*
_output_shapes	
:


fc2/MatMulMatMul
relu1/Relufc2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
{
fc2/BiasAddBiasAdd
fc2/MatMulfc2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
	bn2/ConstConst*
valueB*  ?*
dtype0*
_output_shapes	
:
w
	bn2/gamma
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

bn2/gamma/AssignAssign	bn2/gamma	bn2/Const*
T0*
_class
loc:@bn2/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
i
bn2/gamma/readIdentity	bn2/gamma*
_output_shapes	
:*
T0*
_class
loc:@bn2/gamma
Z
bn2/Const_1Const*
dtype0*
_output_shapes	
:*
valueB*    
v
bn2/beta
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 

bn2/beta/AssignAssignbn2/betabn2/Const_1*
T0*
_class
loc:@bn2/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
f
bn2/beta/readIdentitybn2/beta*
T0*
_class
loc:@bn2/beta*
_output_shapes	
:
Z
bn2/Const_2Const*
valueB*    *
dtype0*
_output_shapes	
:
}
bn2/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
ą
bn2/moving_mean/AssignAssignbn2/moving_meanbn2/Const_2*
use_locking(*
T0*"
_class
loc:@bn2/moving_mean*
validate_shape(*
_output_shapes	
:
{
bn2/moving_mean/readIdentitybn2/moving_mean*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:
Z
bn2/Const_3Const*
valueB*  ?*
dtype0*
_output_shapes	
:

bn2/moving_variance
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
˝
bn2/moving_variance/AssignAssignbn2/moving_variancebn2/Const_3*
use_locking(*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes	
:

bn2/moving_variance/readIdentitybn2/moving_variance*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:
l
"bn2/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 

bn2/moments/meanMeanfc2/BiasAdd"bn2/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
d
bn2/moments/StopGradientStopGradientbn2/moments/mean*
T0*
_output_shapes
:	

bn2/moments/SquaredDifferenceSquaredDifferencefc2/BiasAddbn2/moments/StopGradient*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
&bn2/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ş
bn2/moments/varianceMeanbn2/moments/SquaredDifference&bn2/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
m
bn2/moments/SqueezeSqueezebn2/moments/mean*
T0*
_output_shapes	
:*
squeeze_dims
 
s
bn2/moments/Squeeze_1Squeezebn2/moments/variance*
T0*
_output_shapes	
:*
squeeze_dims
 
X
bn2/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
j
bn2/batchnorm/addAddbn2/moments/Squeeze_1bn2/batchnorm/add/y*
T0*
_output_shapes	
:
U
bn2/batchnorm/RsqrtRsqrtbn2/batchnorm/add*
T0*
_output_shapes	
:
c
bn2/batchnorm/mulMulbn2/batchnorm/Rsqrtbn2/gamma/read*
_output_shapes	
:*
T0
m
bn2/batchnorm/mul_1Mulfc2/BiasAddbn2/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
bn2/batchnorm/mul_2Mulbn2/moments/Squeezebn2/batchnorm/mul*
T0*
_output_shapes	
:
b
bn2/batchnorm/subSubbn2/beta/readbn2/batchnorm/mul_2*
T0*
_output_shapes	
:
u
bn2/batchnorm/add_1Addbn2/batchnorm/mul_1bn2/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
	bn2/ShapeShapefc2/BiasAdd*
T0*
out_type0*
_output_shapes
:
a
bn2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
bn2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
bn2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bn2/strided_sliceStridedSlice	bn2/Shapebn2/strided_slice/stackbn2/strided_slice/stack_1bn2/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
d
bn2/Rank/packedPackbn2/strided_slice*
T0*

axis *
N*
_output_shapes
:
J
bn2/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Q
bn2/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
Q
bn2/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
f
	bn2/rangeRangebn2/range/startbn2/Rankbn2/range/delta*

Tidx0*
_output_shapes
:
c
bn2/Prod/inputPackbn2/strided_slice*
T0*

axis *
N*
_output_shapes
:
i
bn2/ProdProdbn2/Prod/input	bn2/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Z
bn2/CastCastbn2/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
N
	bn2/sub/yConst*
valueB
 *Ĺ ?*
dtype0*
_output_shapes
: 
D
bn2/subSubbn2/Cast	bn2/sub/y*
T0*
_output_shapes
: 
J
bn2/truedivRealDivbn2/Castbn2/sub*
_output_shapes
: *
T0
X
bn2/mulMulbn2/moments/Squeeze_1bn2/truediv*
_output_shapes	
:*
T0

bn2/AssignMovingAvg/decayConst*
valueB
 *
×#<*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 

)bn2/AssignMovingAvg/bn2/moving_mean/zerosConst*
valueB*    *"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes	
:
¨
bn2/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@bn2/moving_mean*
	container *
shape:
Ý
bn2/moving_mean/biased/AssignAssignbn2/moving_mean/biased)bn2/AssignMovingAvg/bn2/moving_mean/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@bn2/moving_mean

bn2/moving_mean/biased/readIdentitybn2/moving_mean/biased*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:

,bn2/moving_mean/local_step/Initializer/zerosConst*"
_class
loc:@bn2/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
˘
bn2/moving_mean/local_step
VariableV2*
shared_name *"
_class
loc:@bn2/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
ă
!bn2/moving_mean/local_step/AssignAssignbn2/moving_mean/local_step,bn2/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@bn2/moving_mean*
validate_shape(*
_output_shapes
: 

bn2/moving_mean/local_step/readIdentitybn2/moving_mean/local_step*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes
: 
Ş
'bn2/AssignMovingAvg/bn2/moving_mean/subSubbn2/moving_mean/biased/readbn2/moments/Squeeze*
_output_shapes	
:*
T0*"
_class
loc:@bn2/moving_mean
ź
'bn2/AssignMovingAvg/bn2/moving_mean/mulMul'bn2/AssignMovingAvg/bn2/moving_mean/subbn2/AssignMovingAvg/decay*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:
Ţ
3bn2/AssignMovingAvg/bn2/moving_mean/bn2/moving_mean	AssignSubbn2/moving_mean/biased'bn2/AssignMovingAvg/bn2/moving_mean/mul*
use_locking( *
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:

3bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd/valueConst*
valueB
 *  ?*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 
ă
-bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd	AssignAddbn2/moving_mean/local_step3bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd/value*
use_locking( *
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes
: 
ü
(bn2/AssignMovingAvg/bn2/moving_mean/readIdentitybn2/moving_mean/biased.^bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd4^bn2/AssignMovingAvg/bn2/moving_mean/bn2/moving_mean*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:
ú
+bn2/AssignMovingAvg/bn2/moving_mean/sub_1/xConst.^bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd4^bn2/AssignMovingAvg/bn2/moving_mean/bn2/moving_mean*
valueB
 *  ?*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 
˝
)bn2/AssignMovingAvg/bn2/moving_mean/sub_1Sub+bn2/AssignMovingAvg/bn2/moving_mean/sub_1/xbn2/AssignMovingAvg/decay*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes
: 
ý
*bn2/AssignMovingAvg/bn2/moving_mean/read_1Identitybn2/moving_mean/local_step.^bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd4^bn2/AssignMovingAvg/bn2/moving_mean/bn2/moving_mean*
_output_shapes
: *
T0*"
_class
loc:@bn2/moving_mean
Ę
'bn2/AssignMovingAvg/bn2/moving_mean/PowPow)bn2/AssignMovingAvg/bn2/moving_mean/sub_1*bn2/AssignMovingAvg/bn2/moving_mean/read_1*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes
: 
ú
+bn2/AssignMovingAvg/bn2/moving_mean/sub_2/xConst.^bn2/AssignMovingAvg/bn2/moving_mean/AssignAdd4^bn2/AssignMovingAvg/bn2/moving_mean/bn2/moving_mean*
dtype0*
_output_shapes
: *
valueB
 *  ?*"
_class
loc:@bn2/moving_mean
Ë
)bn2/AssignMovingAvg/bn2/moving_mean/sub_2Sub+bn2/AssignMovingAvg/bn2/moving_mean/sub_2/x'bn2/AssignMovingAvg/bn2/moving_mean/Pow*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes
: 
Ő
+bn2/AssignMovingAvg/bn2/moving_mean/truedivRealDiv(bn2/AssignMovingAvg/bn2/moving_mean/read)bn2/AssignMovingAvg/bn2/moving_mean/sub_2*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:
˝
)bn2/AssignMovingAvg/bn2/moving_mean/sub_3Subbn2/moving_mean/read+bn2/AssignMovingAvg/bn2/moving_mean/truediv*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:
š
bn2/AssignMovingAvg	AssignSubbn2/moving_mean)bn2/AssignMovingAvg/bn2/moving_mean/sub_3*
T0*"
_class
loc:@bn2/moving_mean*
_output_shapes	
:*
use_locking( 

bn2/AssignMovingAvg_1/decayConst*
valueB
 *
×#<*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 
Ś
/bn2/AssignMovingAvg_1/bn2/moving_variance/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *&
_class
loc:@bn2/moving_variance
°
bn2/moving_variance/biased
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *&
_class
loc:@bn2/moving_variance*
	container 
ď
!bn2/moving_variance/biased/AssignAssignbn2/moving_variance/biased/bn2/AssignMovingAvg_1/bn2/moving_variance/zeros*
use_locking(*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes	
:

bn2/moving_variance/biased/readIdentitybn2/moving_variance/biased*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:

0bn2/moving_variance/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *&
_class
loc:@bn2/moving_variance*
valueB
 *    
Ş
bn2/moving_variance/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *&
_class
loc:@bn2/moving_variance*
	container *
shape: 
ó
%bn2/moving_variance/local_step/AssignAssignbn2/moving_variance/local_step0bn2/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes
: 

#bn2/moving_variance/local_step/readIdentitybn2/moving_variance/local_step*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes
: 
Ź
-bn2/AssignMovingAvg_1/bn2/moving_variance/subSubbn2/moving_variance/biased/readbn2/mul*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:
Î
-bn2/AssignMovingAvg_1/bn2/moving_variance/mulMul-bn2/AssignMovingAvg_1/bn2/moving_variance/subbn2/AssignMovingAvg_1/decay*
_output_shapes	
:*
T0*&
_class
loc:@bn2/moving_variance
ö
=bn2/AssignMovingAvg_1/bn2/moving_variance/bn2/moving_variance	AssignSubbn2/moving_variance/biased-bn2/AssignMovingAvg_1/bn2/moving_variance/mul*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:*
use_locking( 
Ś
9bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd/valueConst*
valueB
 *  ?*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 
÷
3bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd	AssignAddbn2/moving_variance/local_step9bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*&
_class
loc:@bn2/moving_variance

.bn2/AssignMovingAvg_1/bn2/moving_variance/readIdentitybn2/moving_variance/biased4^bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd>^bn2/AssignMovingAvg_1/bn2/moving_variance/bn2/moving_variance*
_output_shapes	
:*
T0*&
_class
loc:@bn2/moving_variance

1bn2/AssignMovingAvg_1/bn2/moving_variance/sub_1/xConst4^bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd>^bn2/AssignMovingAvg_1/bn2/moving_variance/bn2/moving_variance*
valueB
 *  ?*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 
Ď
/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_1Sub1bn2/AssignMovingAvg_1/bn2/moving_variance/sub_1/xbn2/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*&
_class
loc:@bn2/moving_variance

0bn2/AssignMovingAvg_1/bn2/moving_variance/read_1Identitybn2/moving_variance/local_step4^bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd>^bn2/AssignMovingAvg_1/bn2/moving_variance/bn2/moving_variance*
_output_shapes
: *
T0*&
_class
loc:@bn2/moving_variance
ŕ
-bn2/AssignMovingAvg_1/bn2/moving_variance/PowPow/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_10bn2/AssignMovingAvg_1/bn2/moving_variance/read_1*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes
: 

1bn2/AssignMovingAvg_1/bn2/moving_variance/sub_2/xConst4^bn2/AssignMovingAvg_1/bn2/moving_variance/AssignAdd>^bn2/AssignMovingAvg_1/bn2/moving_variance/bn2/moving_variance*
valueB
 *  ?*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 
á
/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_2Sub1bn2/AssignMovingAvg_1/bn2/moving_variance/sub_2/x-bn2/AssignMovingAvg_1/bn2/moving_variance/Pow*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes
: 
ë
1bn2/AssignMovingAvg_1/bn2/moving_variance/truedivRealDiv.bn2/AssignMovingAvg_1/bn2/moving_variance/read/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_2*
_output_shapes	
:*
T0*&
_class
loc:@bn2/moving_variance
Ń
/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_3Subbn2/moving_variance/read1bn2/AssignMovingAvg_1/bn2/moving_variance/truediv*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:
É
bn2/AssignMovingAvg_1	AssignSubbn2/moving_variance/bn2/AssignMovingAvg_1/bn2/moving_variance/sub_3*
T0*&
_class
loc:@bn2/moving_variance*
_output_shapes	
:*
use_locking( 
h
bn2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

Q
bn2/cond/switch_tIdentitybn2/cond/Switch:1*
T0
*
_output_shapes
: 
O
bn2/cond/switch_fIdentitybn2/cond/Switch*
T0
*
_output_shapes
: 
S
bn2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
ą
bn2/cond/Switch_1Switchbn2/batchnorm/add_1bn2/cond/pred_id*
T0*&
_class
loc:@bn2/batchnorm/add_1*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
q
bn2/cond/batchnorm/add/yConst^bn2/cond/switch_f*
valueB
 *o:*
dtype0*
_output_shapes
: 
|
bn2/cond/batchnorm/addAddbn2/cond/batchnorm/add/Switchbn2/cond/batchnorm/add/y*
T0*
_output_shapes	
:
¨
bn2/cond/batchnorm/add/SwitchSwitchbn2/moving_variance/readbn2/cond/pred_id*
T0*&
_class
loc:@bn2/moving_variance*"
_output_shapes
::
_
bn2/cond/batchnorm/RsqrtRsqrtbn2/cond/batchnorm/add*
_output_shapes	
:*
T0
|
bn2/cond/batchnorm/mulMulbn2/cond/batchnorm/Rsqrtbn2/cond/batchnorm/mul/Switch*
T0*
_output_shapes	
:

bn2/cond/batchnorm/mul/SwitchSwitchbn2/gamma/readbn2/cond/pred_id*
T0*
_class
loc:@bn2/gamma*"
_output_shapes
::

bn2/cond/batchnorm/mul_1Mulbn2/cond/batchnorm/mul_1/Switchbn2/cond/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
bn2/cond/batchnorm/mul_1/SwitchSwitchfc2/BiasAddbn2/cond/pred_id*
T0*
_class
loc:@fc2/BiasAdd*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
~
bn2/cond/batchnorm/mul_2Mulbn2/cond/batchnorm/mul_2/Switchbn2/cond/batchnorm/mul*
_output_shapes	
:*
T0
˘
bn2/cond/batchnorm/mul_2/SwitchSwitchbn2/moving_mean/readbn2/cond/pred_id*"
_output_shapes
::*
T0*"
_class
loc:@bn2/moving_mean
|
bn2/cond/batchnorm/subSubbn2/cond/batchnorm/sub/Switchbn2/cond/batchnorm/mul_2*
T0*
_output_shapes	
:

bn2/cond/batchnorm/sub/SwitchSwitchbn2/beta/readbn2/cond/pred_id*
T0*
_class
loc:@bn2/beta*"
_output_shapes
::

bn2/cond/batchnorm/add_1Addbn2/cond/batchnorm/mul_1bn2/cond/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

bn2/cond/MergeMergebn2/cond/batchnorm/add_1bn2/cond/Switch_1:1*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0
U

relu2/ReluRelubn2/cond/Merge*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
output/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
^
output/random_uniform/minConst*
valueB
 *n×\ž*
dtype0*
_output_shapes
: 
^
output/random_uniform/maxConst*
valueB
 *n×\>*
dtype0*
_output_shapes
: 
§
#output/random_uniform/RandomUniformRandomUniformoutput/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*
_output_shapes
:	*
seed2żĂ
w
output/random_uniform/subSuboutput/random_uniform/maxoutput/random_uniform/min*
T0*
_output_shapes
: 

output/random_uniform/mulMul#output/random_uniform/RandomUniformoutput/random_uniform/sub*
_output_shapes
:	*
T0
|
output/random_uniformAddoutput/random_uniform/muloutput/random_uniform/min*
T0*
_output_shapes
:	

output/kernel
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
š
output/kernel/AssignAssignoutput/kerneloutput/random_uniform*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
y
output/kernel/readIdentityoutput/kernel*
T0* 
_class
loc:@output/kernel*
_output_shapes
:	
Y
output/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
w
output/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ľ
output/bias/AssignAssignoutput/biasoutput/Const*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:*
use_locking(
n
output/bias/readIdentityoutput/bias*
_output_shapes
:*
T0*
_class
loc:@output/bias

output/MatMulMatMul
relu2/Reluoutput/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

output/BiasAddBiasAddoutput/MatMuloutput/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
ş
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*!
_class
loc:@SGD/iterations
s
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 

SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/lr
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
˛
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
m
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ś
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: *
use_locking(
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 

output_targetPlaceholder*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
output_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
u
loss/output_loss/subSuboutput/BiasAddoutput_target*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
r
loss/output_loss/SquareSquareloss/output_loss/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
r
'loss/output_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ş
loss/output_loss/MeanMeanloss/output_loss/Square'loss/output_loss/Mean/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
l
)loss/output_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Ź
loss/output_loss/Mean_1Meanloss/output_loss/Mean)loss/output_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
y
loss/output_loss/mulMulloss/output_loss/Mean_1output_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
loss/output_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_loss/NotEqualNotEqualoutput_sample_weightsloss/output_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

loss/output_loss/CastCastloss/output_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
`
loss/output_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_2Meanloss/output_loss/Castloss/output_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

loss/output_loss/truedivRealDivloss/output_loss/mulloss/output_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
loss/output_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_3Meanloss/output_loss/truedivloss/output_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_loss/Mean_3*
T0*
_output_shapes
: 
q
metrics/rmse/subSuboutput_targetoutput/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
metrics/rmse/Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
x
metrics/rmse/PowPowmetrics/rmse/submetrics/rmse/Pow/y*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
c
metrics/rmse/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
}
metrics/rmse/MeanMeanmetrics/rmse/Powmetrics/rmse/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Y
metrics/rmse/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
Y
metrics/rmse/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  
w
"metrics/rmse/clip_by_value/MinimumMinimummetrics/rmse/Meanmetrics/rmse/Const_2*
T0*
_output_shapes
: 

metrics/rmse/clip_by_valueMaximum"metrics/rmse/clip_by_value/Minimummetrics/rmse/Const_1*
T0*
_output_shapes
: 
V
metrics/rmse/SqrtSqrtmetrics/rmse/clip_by_value*
T0*
_output_shapes
: 
W
metrics/rmse/Const_3Const*
valueB *
dtype0*
_output_shapes
: 

metrics/rmse/Mean_1Meanmetrics/rmse/Sqrtmetrics/rmse/Const_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
|
training/SGD/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@loss/mul

 training/SGD/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@loss/mul
ł
training/SGD/gradients/FillFilltraining/SGD/gradients/Shape training/SGD/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
Ł
(training/SGD/gradients/loss/mul_grad/MulMultraining/SGD/gradients/Fillloss/output_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

*training/SGD/gradients/loss/mul_grad/Mul_1Multraining/SGD/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ˇ
Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shapeConst*
valueB:**
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
:

;training/SGD/gradients/loss/output_loss/Mean_3_grad/ReshapeReshape*training/SGD/gradients/loss/mul_grad/Mul_1Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
˝
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ShapeShapeloss/output_loss/truediv*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
¤
8training/SGD/gradients/loss/output_loss/Mean_3_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape9training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_1Shapeloss/output_loss/truediv*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3
Ş
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2Const*
valueB **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
: 
Ż
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: **
_class 
loc:@loss/output_loss/Mean_3
˘
8training/SGD/gradients/loss/output_loss/Mean_3_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_19training/SGD/gradients/loss/output_loss/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_3
ą
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1Const*
valueB: **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
:
Ś
:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ť
=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_3_grad/MaximumMaximum:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordivFloorDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0**
_class 
loc:@loss/output_loss/Mean_3
ę
8training/SGD/gradients/loss/output_loss/Mean_3_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/output_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0

;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_3_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
:training/SGD/gradients/loss/output_loss/truediv_grad/ShapeShapeloss/output_loss/mul*
T0*
out_type0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:
Ź
<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1Const*
valueB *+
_class!
loc:@loss/output_loss/truediv*
dtype0*
_output_shapes
: 
Ç
Jtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/loss/output_loss/truediv_grad/Shape<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/output_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ř
<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivRealDiv;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
8training/SGD/gradients/loss/output_loss/truediv_grad/SumSum<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivJtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/output_loss/truediv
Ś
<training/SGD/gradients/loss/output_loss/truediv_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/truediv_grad/Sum:training/SGD/gradients/loss/output_loss/truediv_grad/Shape*
T0*
Tshape0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
8training/SGD/gradients/loss/output_loss/truediv_grad/NegNegloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1RealDiv8training/SGD/gradients/loss/output_loss/truediv_grad/Negloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2RealDiv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1loss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

8training/SGD/gradients/loss/output_loss/truediv_grad/mulMul;training/SGD/gradients/loss/output_loss/Mean_3_grad/truediv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1Sum8training/SGD/gradients/loss/output_loss/truediv_grad/mulLtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/output_loss/truediv

>training/SGD/gradients/loss/output_loss/truediv_grad/Reshape_1Reshape:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
: 
ś
6training/SGD/gradients/loss/output_loss/mul_grad/ShapeShapeloss/output_loss/Mean_1*
T0*
out_type0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:
ś
8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1Shapeoutput_sample_weights*
T0*
out_type0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:
ˇ
Ftraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/mul_grad/Shape8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ç
4training/SGD/gradients/loss/output_loss/mul_grad/MulMul<training/SGD/gradients/loss/output_loss/truediv_grad/Reshapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
4training/SGD/gradients/loss/output_loss/mul_grad/SumSum4training/SGD/gradients/loss/output_loss/mul_grad/MulFtraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/mul_grad/Sum6training/SGD/gradients/loss/output_loss/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Mulloss/output_loss/Mean_1<training/SGD/gradients/loss/output_loss/truediv_grad/Reshape*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
6training/SGD/gradients/loss/output_loss/mul_grad/Sum_1Sum6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Htraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

:training/SGD/gradients/loss/output_loss/mul_grad/Reshape_1Reshape6training/SGD/gradients/loss/output_loss/mul_grad/Sum_18training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ShapeShapeloss/output_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Ś
8training/SGD/gradients/loss/output_loss/Mean_1_grad/SizeConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
ň
7training/SGD/gradients/loss/output_loss/Mean_1_grad/addAdd)loss/output_loss/Mean_1/reduction_indices8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

7training/SGD/gradients/loss/output_loss/Mean_1_grad/modFloorMod7training/SGD/gradients/loss/output_loss/Mean_1_grad/add8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
ą
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1Const*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
Ň
9training/SGD/gradients/loss/output_loss/Mean_1_grad/rangeRange?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/start8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/delta*
_output_shapes
:*

Tidx0**
_class 
loc:@loss/output_loss/Mean_1
Ź
>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/output_loss/Mean_1

8training/SGD/gradients/loss/output_loss/Mean_1_grad/FillFill;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

Atraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitchDynamicStitch9training/SGD/gradients/loss/output_loss/Mean_1_grad/range7training/SGD/gradients/loss/output_loss/Mean_1_grad/mod9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape8training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill*
T0**
_class 
loc:@loss/output_loss/Mean_1*
N*
_output_shapes
:
Ť
=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_1_grad/MaximumMaximumAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1

<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordivFloorDiv9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape;training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Ť
;training/SGD/gradients/loss/output_loss/Mean_1_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_1
§
8training/SGD/gradients/loss/output_loss/Mean_1_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_1_grad/Reshape<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_1
ź
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_2Shapeloss/output_loss/Mean*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1
ž
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3Shapeloss/output_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Ż
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ConstConst*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
˘
8training/SGD/gradients/loss/output_loss/Mean_1_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_29training/SGD/gradients/loss/output_loss/Mean_1_grad/Const*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
ą
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: **
_class 
loc:@loss/output_loss/Mean_1
Ś
:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/output_loss/Mean_1

=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1Maximum:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/y*
_output_shapes
: *
T0**
_class 
loc:@loss/output_loss/Mean_1

>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1FloorDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
ě
8training/SGD/gradients/loss/output_loss/Mean_1_grad/CastCast>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/output_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0

;training/SGD/gradients/loss/output_loss/Mean_1_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_1_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
7training/SGD/gradients/loss/output_loss/Mean_grad/ShapeShapeloss/output_loss/Square*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
˘
6training/SGD/gradients/loss/output_loss/Mean_grad/SizeConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
č
5training/SGD/gradients/loss/output_loss/Mean_grad/addAdd'loss/output_loss/Mean/reduction_indices6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/output_loss/Mean
ű
5training/SGD/gradients/loss/output_loss/Mean_grad/modFloorMod5training/SGD/gradients/loss/output_loss/Mean_grad/add6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
Ś
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1Const*
valueB *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Š
=training/SGD/gradients/loss/output_loss/Mean_grad/range/startConst*
value	B : *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Š
=training/SGD/gradients/loss/output_loss/Mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*(
_class
loc:@loss/output_loss/Mean
Č
7training/SGD/gradients/loss/output_loss/Mean_grad/rangeRange=training/SGD/gradients/loss/output_loss/Mean_grad/range/start6training/SGD/gradients/loss/output_loss/Mean_grad/Size=training/SGD/gradients/loss/output_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*(
_class
loc:@loss/output_loss/Mean
¨
<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

6training/SGD/gradients/loss/output_loss/Mean_grad/FillFill9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/value*
T0*

index_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitchDynamicStitch7training/SGD/gradients/loss/output_loss/Mean_grad/range5training/SGD/gradients/loss/output_loss/Mean_grad/mod7training/SGD/gradients/loss/output_loss/Mean_grad/Shape6training/SGD/gradients/loss/output_loss/Mean_grad/Fill*
T0*(
_class
loc:@loss/output_loss/Mean*
N*
_output_shapes
:
§
;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/yConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

9training/SGD/gradients/loss/output_loss/Mean_grad/MaximumMaximum?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:

:training/SGD/gradients/loss/output_loss/Mean_grad/floordivFloorDiv7training/SGD/gradients/loss/output_loss/Mean_grad/Shape9training/SGD/gradients/loss/output_loss/Mean_grad/Maximum*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
ľ
9training/SGD/gradients/loss/output_loss/Mean_grad/ReshapeReshape;training/SGD/gradients/loss/output_loss/Mean_1_grad/truediv?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*(
_class
loc:@loss/output_loss/Mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ź
6training/SGD/gradients/loss/output_loss/Mean_grad/TileTile9training/SGD/gradients/loss/output_loss/Mean_grad/Reshape:training/SGD/gradients/loss/output_loss/Mean_grad/floordiv*

Tmultiples0*
T0*(
_class
loc:@loss/output_loss/Mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ş
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_2Shapeloss/output_loss/Square*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
¸
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_3Shapeloss/output_loss/Mean*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ť
7training/SGD/gradients/loss/output_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *(
_class
loc:@loss/output_loss/Mean

6training/SGD/gradients/loss/output_loss/Mean_grad/ProdProd9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_27training/SGD/gradients/loss/output_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
­
9training/SGD/gradients/loss/output_loss/Mean_grad/Const_1Const*
valueB: *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1Prod9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_39training/SGD/gradients/loss/output_loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
Š
=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/yConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1Maximum8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1FloorDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ć
6training/SGD/gradients/loss/output_loss/Mean_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*(
_class
loc:@loss/output_loss/Mean

9training/SGD/gradients/loss/output_loss/Mean_grad/truedivRealDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Tile6training/SGD/gradients/loss/output_loss/Mean_grad/Cast*
T0*(
_class
loc:@loss/output_loss/Mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ć
9training/SGD/gradients/loss/output_loss/Square_grad/ConstConst:^training/SGD/gradients/loss/output_loss/Mean_grad/truediv*
valueB
 *   @**
_class 
loc:@loss/output_loss/Square*
dtype0*
_output_shapes
: 
ö
7training/SGD/gradients/loss/output_loss/Square_grad/MulMulloss/output_loss/sub9training/SGD/gradients/loss/output_loss/Square_grad/Const*
T0**
_class 
loc:@loss/output_loss/Square*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

9training/SGD/gradients/loss/output_loss/Square_grad/Mul_1Mul9training/SGD/gradients/loss/output_loss/Mean_grad/truediv7training/SGD/gradients/loss/output_loss/Square_grad/Mul*
T0**
_class 
loc:@loss/output_loss/Square*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
6training/SGD/gradients/loss/output_loss/sub_grad/ShapeShapeoutput/BiasAdd*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@loss/output_loss/sub
Ž
8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1Shapeoutput_target*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@loss/output_loss/sub
ˇ
Ftraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/sub_grad/Shape8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
§
4training/SGD/gradients/loss/output_loss/sub_grad/SumSum9training/SGD/gradients/loss/output_loss/Square_grad/Mul_1Ftraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/sub

8training/SGD/gradients/loss/output_loss/sub_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/sub_grad/Sum6training/SGD/gradients/loss/output_loss/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub
Ť
6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1Sum9training/SGD/gradients/loss/output_loss/Square_grad/Mul_1Htraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:
ż
4training/SGD/gradients/loss/output_loss/sub_grad/NegNeg6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1*
_output_shapes
:*
T0*'
_class
loc:@loss/output_loss/sub
§
:training/SGD/gradients/loss/output_loss/sub_grad/Reshape_1Reshape4training/SGD/gradients/loss/output_loss/sub_grad/Neg8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub
Ţ
6training/SGD/gradients/output/BiasAdd_grad/BiasAddGradBiasAddGrad8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*!
_class
loc:@output/BiasAdd

0training/SGD/gradients/output/MatMul_grad/MatMulMatMul8training/SGD/gradients/loss/output_loss/sub_grad/Reshapeoutput/kernel/read*
T0* 
_class
loc:@output/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ô
2training/SGD/gradients/output/MatMul_grad/MatMul_1MatMul
relu2/Relu8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0* 
_class
loc:@output/MatMul
Ë
/training/SGD/gradients/relu2/Relu_grad/ReluGradReluGrad0training/SGD/gradients/output/MatMul_grad/MatMul
relu2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@relu2/Relu
ç
4training/SGD/gradients/bn2/cond/Merge_grad/cond_gradSwitch/training/SGD/gradients/relu2/Relu_grad/ReluGradbn2/cond/pred_id*
T0*
_class
loc:@relu2/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ż
:training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/ShapeShapebn2/cond/batchnorm/mul_1*
T0*
out_type0*+
_class!
loc:@bn2/cond/batchnorm/add_1*
_output_shapes
:
´
<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*+
_class!
loc:@bn2/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ç
Jtraining/SGD/gradients/bn2/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Shape<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Shape_1*
T0*+
_class!
loc:@bn2/cond/batchnorm/add_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
8training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/SumSum4training/SGD/gradients/bn2/cond/Merge_grad/cond_gradJtraining/SGD/gradients/bn2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*+
_class!
loc:@bn2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ť
<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/ReshapeReshape8training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Sum:training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*+
_class!
loc:@bn2/cond/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
:training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Sum_1Sum4training/SGD/gradients/bn2/cond/Merge_grad/cond_gradLtraining/SGD/gradients/bn2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@bn2/cond/batchnorm/add_1*
_output_shapes
:
¤
>training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Reshape_1Reshape:training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Sum_1<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@bn2/cond/batchnorm/add_1*
_output_shapes	
:
˝
training/SGD/gradients/SwitchSwitchbn2/batchnorm/add_1bn2/cond/pred_id*
T0*&
_class
loc:@bn2/batchnorm/add_1*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
training/SGD/gradients/IdentityIdentitytraining/SGD/gradients/Switch*
T0*&
_class
loc:@bn2/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
training/SGD/gradients/Shape_1Shapetraining/SGD/gradients/Switch*
T0*
out_type0*&
_class
loc:@bn2/batchnorm/add_1*
_output_shapes
:
ą
"training/SGD/gradients/zeros/ConstConst ^training/SGD/gradients/Identity*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@bn2/batchnorm/add_1
Ő
training/SGD/gradients/zerosFilltraining/SGD/gradients/Shape_1"training/SGD/gradients/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0*&
_class
loc:@bn2/batchnorm/add_1
ü
7training/SGD/gradients/bn2/cond/Switch_1_grad/cond_gradMergetraining/SGD/gradients/zeros6training/SGD/gradients/bn2/cond/Merge_grad/cond_grad:1*
T0*&
_class
loc:@bn2/batchnorm/add_1*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Ć
:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/ShapeShapebn2/cond/batchnorm/mul_1/Switch*
T0*
out_type0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*
_output_shapes
:
´
<training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*+
_class!
loc:@bn2/cond/batchnorm/mul_1
Ç
Jtraining/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Shape<training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Shape_1*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
8training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/MulMul<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Reshapebn2/cond/batchnorm/mul*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
8training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/SumSum8training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/MulJtraining/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*
_output_shapes
:
Ť
<training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/ReshapeReshape8training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Sum:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Mul_1Mulbn2/cond/batchnorm/mul_1/Switch<training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Reshape*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Sum_1Sum:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Mul_1Ltraining/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1
¤
>training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Reshape_1Reshape:training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Sum_1<training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*
_output_shapes	
:
Î
6training/SGD/gradients/bn2/cond/batchnorm/sub_grad/NegNeg>training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Reshape_1*
T0*)
_class
loc:@bn2/cond/batchnorm/sub*
_output_shapes	
:
°
5training/SGD/gradients/bn2/batchnorm/add_1_grad/ShapeShapebn2/batchnorm/mul_1*
T0*
out_type0*&
_class
loc:@bn2/batchnorm/add_1*
_output_shapes
:
Ş
7training/SGD/gradients/bn2/batchnorm/add_1_grad/Shape_1Const*
valueB:*&
_class
loc:@bn2/batchnorm/add_1*
dtype0*
_output_shapes
:
ł
Etraining/SGD/gradients/bn2/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/SGD/gradients/bn2/batchnorm/add_1_grad/Shape7training/SGD/gradients/bn2/batchnorm/add_1_grad/Shape_1*
T0*&
_class
loc:@bn2/batchnorm/add_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˘
3training/SGD/gradients/bn2/batchnorm/add_1_grad/SumSum7training/SGD/gradients/bn2/cond/Switch_1_grad/cond_gradEtraining/SGD/gradients/bn2/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*&
_class
loc:@bn2/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 

7training/SGD/gradients/bn2/batchnorm/add_1_grad/ReshapeReshape3training/SGD/gradients/bn2/batchnorm/add_1_grad/Sum5training/SGD/gradients/bn2/batchnorm/add_1_grad/Shape*
T0*
Tshape0*&
_class
loc:@bn2/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
5training/SGD/gradients/bn2/batchnorm/add_1_grad/Sum_1Sum7training/SGD/gradients/bn2/cond/Switch_1_grad/cond_gradGtraining/SGD/gradients/bn2/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*&
_class
loc:@bn2/batchnorm/add_1*
_output_shapes
:

9training/SGD/gradients/bn2/batchnorm/add_1_grad/Reshape_1Reshape5training/SGD/gradients/bn2/batchnorm/add_1_grad/Sum_17training/SGD/gradients/bn2/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*&
_class
loc:@bn2/batchnorm/add_1*
_output_shapes	
:
Ż
training/SGD/gradients/Switch_1Switchfc2/BiasAddbn2/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@fc2/BiasAdd
Ł
!training/SGD/gradients/Identity_1Identity!training/SGD/gradients/Switch_1:1*
T0*
_class
loc:@fc2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

training/SGD/gradients/Shape_2Shape!training/SGD/gradients/Switch_1:1*
_output_shapes
:*
T0*
out_type0*
_class
loc:@fc2/BiasAdd
­
$training/SGD/gradients/zeros_1/ConstConst"^training/SGD/gradients/Identity_1*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@fc2/BiasAdd
Ń
training/SGD/gradients/zeros_1Filltraining/SGD/gradients/Shape_2$training/SGD/gradients/zeros_1/Const*
T0*

index_type0*
_class
loc:@fc2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Etraining/SGD/gradients/bn2/cond/batchnorm/mul_1/Switch_grad/cond_gradMerge<training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Reshapetraining/SGD/gradients/zeros_1*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*
_class
loc:@fc2/BiasAdd

training/SGD/gradients/Switch_2Switchbn2/beta/readbn2/cond/pred_id*"
_output_shapes
::*
T0*
_class
loc:@bn2/beta

!training/SGD/gradients/Identity_2Identity!training/SGD/gradients/Switch_2:1*
_output_shapes	
:*
T0*
_class
loc:@bn2/beta

training/SGD/gradients/Shape_3Shape!training/SGD/gradients/Switch_2:1*
T0*
out_type0*
_class
loc:@bn2/beta*
_output_shapes
:
Ş
$training/SGD/gradients/zeros_2/ConstConst"^training/SGD/gradients/Identity_2*
valueB
 *    *
_class
loc:@bn2/beta*
dtype0*
_output_shapes
: 
Á
training/SGD/gradients/zeros_2Filltraining/SGD/gradients/Shape_3$training/SGD/gradients/zeros_2/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@bn2/beta
ú
Ctraining/SGD/gradients/bn2/cond/batchnorm/sub/Switch_grad/cond_gradMerge>training/SGD/gradients/bn2/cond/batchnorm/add_1_grad/Reshape_1training/SGD/gradients/zeros_2*
T0*
_class
loc:@bn2/beta*
N*
_output_shapes
	:: 
â
8training/SGD/gradients/bn2/cond/batchnorm/mul_2_grad/MulMul6training/SGD/gradients/bn2/cond/batchnorm/sub_grad/Negbn2/cond/batchnorm/mul*
_output_shapes	
:*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_2
í
:training/SGD/gradients/bn2/cond/batchnorm/mul_2_grad/Mul_1Mul6training/SGD/gradients/bn2/cond/batchnorm/sub_grad/Negbn2/cond/batchnorm/mul_2/Switch*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_2*
_output_shapes	
:
¨
5training/SGD/gradients/bn2/batchnorm/mul_1_grad/ShapeShapefc2/BiasAdd*
_output_shapes
:*
T0*
out_type0*&
_class
loc:@bn2/batchnorm/mul_1
Ş
7training/SGD/gradients/bn2/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@bn2/batchnorm/mul_1
ł
Etraining/SGD/gradients/bn2/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Shape7training/SGD/gradients/bn2/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*&
_class
loc:@bn2/batchnorm/mul_1
á
3training/SGD/gradients/bn2/batchnorm/mul_1_grad/MulMul7training/SGD/gradients/bn2/batchnorm/add_1_grad/Reshapebn2/batchnorm/mul*
T0*&
_class
loc:@bn2/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

3training/SGD/gradients/bn2/batchnorm/mul_1_grad/SumSum3training/SGD/gradients/bn2/batchnorm/mul_1_grad/MulEtraining/SGD/gradients/bn2/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*&
_class
loc:@bn2/batchnorm/mul_1*
_output_shapes
:

7training/SGD/gradients/bn2/batchnorm/mul_1_grad/ReshapeReshape3training/SGD/gradients/bn2/batchnorm/mul_1_grad/Sum5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*&
_class
loc:@bn2/batchnorm/mul_1
Ý
5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Mul_1Mulfc2/BiasAdd7training/SGD/gradients/bn2/batchnorm/add_1_grad/Reshape*
T0*&
_class
loc:@bn2/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Sum_1Sum5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Mul_1Gtraining/SGD/gradients/bn2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*&
_class
loc:@bn2/batchnorm/mul_1

9training/SGD/gradients/bn2/batchnorm/mul_1_grad/Reshape_1Reshape5training/SGD/gradients/bn2/batchnorm/mul_1_grad/Sum_17training/SGD/gradients/bn2/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0*&
_class
loc:@bn2/batchnorm/mul_1
ż
1training/SGD/gradients/bn2/batchnorm/sub_grad/NegNeg9training/SGD/gradients/bn2/batchnorm/add_1_grad/Reshape_1*
T0*$
_class
loc:@bn2/batchnorm/sub*
_output_shapes	
:
ű
training/SGD/gradients/AddNAddN>training/SGD/gradients/bn2/cond/batchnorm/mul_1_grad/Reshape_1:training/SGD/gradients/bn2/cond/batchnorm/mul_2_grad/Mul_1*
T0*+
_class!
loc:@bn2/cond/batchnorm/mul_1*
N*
_output_shapes	
:
Ę
6training/SGD/gradients/bn2/cond/batchnorm/mul_grad/MulMultraining/SGD/gradients/AddNbn2/cond/batchnorm/mul/Switch*
T0*)
_class
loc:@bn2/cond/batchnorm/mul*
_output_shapes	
:
Ç
8training/SGD/gradients/bn2/cond/batchnorm/mul_grad/Mul_1Multraining/SGD/gradients/AddNbn2/cond/batchnorm/Rsqrt*
T0*)
_class
loc:@bn2/cond/batchnorm/mul*
_output_shapes	
:
ń
training/SGD/gradients/AddN_1AddNCtraining/SGD/gradients/bn2/cond/batchnorm/sub/Switch_grad/cond_grad9training/SGD/gradients/bn2/batchnorm/add_1_grad/Reshape_1*
T0*
_class
loc:@bn2/beta*
N*
_output_shapes	
:
Î
3training/SGD/gradients/bn2/batchnorm/mul_2_grad/MulMul1training/SGD/gradients/bn2/batchnorm/sub_grad/Negbn2/batchnorm/mul*
_output_shapes	
:*
T0*&
_class
loc:@bn2/batchnorm/mul_2
Ň
5training/SGD/gradients/bn2/batchnorm/mul_2_grad/Mul_1Mul1training/SGD/gradients/bn2/batchnorm/sub_grad/Negbn2/moments/Squeeze*
T0*&
_class
loc:@bn2/batchnorm/mul_2*
_output_shapes	
:

training/SGD/gradients/Switch_3Switchbn2/gamma/readbn2/cond/pred_id*"
_output_shapes
::*
T0*
_class
loc:@bn2/gamma

!training/SGD/gradients/Identity_3Identity!training/SGD/gradients/Switch_3:1*
T0*
_class
loc:@bn2/gamma*
_output_shapes	
:

training/SGD/gradients/Shape_4Shape!training/SGD/gradients/Switch_3:1*
T0*
out_type0*
_class
loc:@bn2/gamma*
_output_shapes
:
Ť
$training/SGD/gradients/zeros_3/ConstConst"^training/SGD/gradients/Identity_3*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@bn2/gamma
Â
training/SGD/gradients/zeros_3Filltraining/SGD/gradients/Shape_4$training/SGD/gradients/zeros_3/Const*
T0*

index_type0*
_class
loc:@bn2/gamma*
_output_shapes	
:
ő
Ctraining/SGD/gradients/bn2/cond/batchnorm/mul/Switch_grad/cond_gradMerge8training/SGD/gradients/bn2/cond/batchnorm/mul_grad/Mul_1training/SGD/gradients/zeros_3*
T0*
_class
loc:@bn2/gamma*
N*
_output_shapes
	:: 
Ž
5training/SGD/gradients/bn2/moments/Squeeze_grad/ShapeConst*
valueB"      *&
_class
loc:@bn2/moments/Squeeze*
dtype0*
_output_shapes
:

7training/SGD/gradients/bn2/moments/Squeeze_grad/ReshapeReshape3training/SGD/gradients/bn2/batchnorm/mul_2_grad/Mul5training/SGD/gradients/bn2/moments/Squeeze_grad/Shape*
T0*
Tshape0*&
_class
loc:@bn2/moments/Squeeze*
_output_shapes
:	
î
training/SGD/gradients/AddN_2AddN9training/SGD/gradients/bn2/batchnorm/mul_1_grad/Reshape_15training/SGD/gradients/bn2/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:*
T0*&
_class
loc:@bn2/batchnorm/mul_1
ł
1training/SGD/gradients/bn2/batchnorm/mul_grad/MulMultraining/SGD/gradients/AddN_2bn2/gamma/read*
T0*$
_class
loc:@bn2/batchnorm/mul*
_output_shapes	
:
ş
3training/SGD/gradients/bn2/batchnorm/mul_grad/Mul_1Multraining/SGD/gradients/AddN_2bn2/batchnorm/Rsqrt*
T0*$
_class
loc:@bn2/batchnorm/mul*
_output_shapes	
:
Ü
9training/SGD/gradients/bn2/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGradbn2/batchnorm/Rsqrt1training/SGD/gradients/bn2/batchnorm/mul_grad/Mul*
_output_shapes	
:*
T0*&
_class
loc:@bn2/batchnorm/Rsqrt
ě
training/SGD/gradients/AddN_3AddNCtraining/SGD/gradients/bn2/cond/batchnorm/mul/Switch_grad/cond_grad3training/SGD/gradients/bn2/batchnorm/mul_grad/Mul_1*
T0*
_class
loc:@bn2/gamma*
N*
_output_shapes	
:
¤
3training/SGD/gradients/bn2/batchnorm/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*$
_class
loc:@bn2/batchnorm/add

5training/SGD/gradients/bn2/batchnorm/add_grad/Shape_1Const*
valueB *$
_class
loc:@bn2/batchnorm/add*
dtype0*
_output_shapes
: 
Ť
Ctraining/SGD/gradients/bn2/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs3training/SGD/gradients/bn2/batchnorm/add_grad/Shape5training/SGD/gradients/bn2/batchnorm/add_grad/Shape_1*
T0*$
_class
loc:@bn2/batchnorm/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
1training/SGD/gradients/bn2/batchnorm/add_grad/SumSum9training/SGD/gradients/bn2/batchnorm/Rsqrt_grad/RsqrtGradCtraining/SGD/gradients/bn2/batchnorm/add_grad/BroadcastGradientArgs*
T0*$
_class
loc:@bn2/batchnorm/add*
_output_shapes	
:*

Tidx0*
	keep_dims( 

5training/SGD/gradients/bn2/batchnorm/add_grad/ReshapeReshape1training/SGD/gradients/bn2/batchnorm/add_grad/Sum3training/SGD/gradients/bn2/batchnorm/add_grad/Shape*
T0*
Tshape0*$
_class
loc:@bn2/batchnorm/add*
_output_shapes	
:
 
3training/SGD/gradients/bn2/batchnorm/add_grad/Sum_1Sum9training/SGD/gradients/bn2/batchnorm/Rsqrt_grad/RsqrtGradEtraining/SGD/gradients/bn2/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*$
_class
loc:@bn2/batchnorm/add*
_output_shapes
: *

Tidx0*
	keep_dims( 

7training/SGD/gradients/bn2/batchnorm/add_grad/Reshape_1Reshape3training/SGD/gradients/bn2/batchnorm/add_grad/Sum_15training/SGD/gradients/bn2/batchnorm/add_grad/Shape_1*
T0*
Tshape0*$
_class
loc:@bn2/batchnorm/add*
_output_shapes
: 
˛
7training/SGD/gradients/bn2/moments/Squeeze_1_grad/ShapeConst*
valueB"      *(
_class
loc:@bn2/moments/Squeeze_1*
dtype0*
_output_shapes
:

9training/SGD/gradients/bn2/moments/Squeeze_1_grad/ReshapeReshape5training/SGD/gradients/bn2/batchnorm/add_grad/Reshape7training/SGD/gradients/bn2/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*(
_class
loc:@bn2/moments/Squeeze_1*
_output_shapes
:	
ź
6training/SGD/gradients/bn2/moments/variance_grad/ShapeShapebn2/moments/SquaredDifference*
T0*
out_type0*'
_class
loc:@bn2/moments/variance*
_output_shapes
:
 
5training/SGD/gradients/bn2/moments/variance_grad/SizeConst*
value	B :*'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
: 
č
4training/SGD/gradients/bn2/moments/variance_grad/addAdd&bn2/moments/variance/reduction_indices5training/SGD/gradients/bn2/moments/variance_grad/Size*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
:
ű
4training/SGD/gradients/bn2/moments/variance_grad/modFloorMod4training/SGD/gradients/bn2/moments/variance_grad/add5training/SGD/gradients/bn2/moments/variance_grad/Size*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
:
Ť
8training/SGD/gradients/bn2/moments/variance_grad/Shape_1Const*
valueB:*'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
:
§
<training/SGD/gradients/bn2/moments/variance_grad/range/startConst*
value	B : *'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
: 
§
<training/SGD/gradients/bn2/moments/variance_grad/range/deltaConst*
value	B :*'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
: 
Ă
6training/SGD/gradients/bn2/moments/variance_grad/rangeRange<training/SGD/gradients/bn2/moments/variance_grad/range/start5training/SGD/gradients/bn2/moments/variance_grad/Size<training/SGD/gradients/bn2/moments/variance_grad/range/delta*'
_class
loc:@bn2/moments/variance*
_output_shapes
:*

Tidx0
Ś
;training/SGD/gradients/bn2/moments/variance_grad/Fill/valueConst*
value	B :*'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
: 

5training/SGD/gradients/bn2/moments/variance_grad/FillFill8training/SGD/gradients/bn2/moments/variance_grad/Shape_1;training/SGD/gradients/bn2/moments/variance_grad/Fill/value*
T0*

index_type0*'
_class
loc:@bn2/moments/variance*
_output_shapes
:

>training/SGD/gradients/bn2/moments/variance_grad/DynamicStitchDynamicStitch6training/SGD/gradients/bn2/moments/variance_grad/range4training/SGD/gradients/bn2/moments/variance_grad/mod6training/SGD/gradients/bn2/moments/variance_grad/Shape5training/SGD/gradients/bn2/moments/variance_grad/Fill*
N*
_output_shapes
:*
T0*'
_class
loc:@bn2/moments/variance
Ľ
:training/SGD/gradients/bn2/moments/variance_grad/Maximum/yConst*
value	B :*'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
: 

8training/SGD/gradients/bn2/moments/variance_grad/MaximumMaximum>training/SGD/gradients/bn2/moments/variance_grad/DynamicStitch:training/SGD/gradients/bn2/moments/variance_grad/Maximum/y*
_output_shapes
:*
T0*'
_class
loc:@bn2/moments/variance

9training/SGD/gradients/bn2/moments/variance_grad/floordivFloorDiv6training/SGD/gradients/bn2/moments/variance_grad/Shape8training/SGD/gradients/bn2/moments/variance_grad/Maximum*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
:
°
8training/SGD/gradients/bn2/moments/variance_grad/ReshapeReshape9training/SGD/gradients/bn2/moments/Squeeze_1_grad/Reshape>training/SGD/gradients/bn2/moments/variance_grad/DynamicStitch*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*'
_class
loc:@bn2/moments/variance
¨
5training/SGD/gradients/bn2/moments/variance_grad/TileTile8training/SGD/gradients/bn2/moments/variance_grad/Reshape9training/SGD/gradients/bn2/moments/variance_grad/floordiv*

Tmultiples0*
T0*'
_class
loc:@bn2/moments/variance*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ž
8training/SGD/gradients/bn2/moments/variance_grad/Shape_2Shapebn2/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@bn2/moments/variance
˛
8training/SGD/gradients/bn2/moments/variance_grad/Shape_3Const*
valueB"      *'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
:
Š
6training/SGD/gradients/bn2/moments/variance_grad/ConstConst*
valueB: *'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
:

5training/SGD/gradients/bn2/moments/variance_grad/ProdProd8training/SGD/gradients/bn2/moments/variance_grad/Shape_26training/SGD/gradients/bn2/moments/variance_grad/Const*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
: 
Ť
8training/SGD/gradients/bn2/moments/variance_grad/Const_1Const*
valueB: *'
_class
loc:@bn2/moments/variance*
dtype0*
_output_shapes
:

7training/SGD/gradients/bn2/moments/variance_grad/Prod_1Prod8training/SGD/gradients/bn2/moments/variance_grad/Shape_38training/SGD/gradients/bn2/moments/variance_grad/Const_1*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
§
<training/SGD/gradients/bn2/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@bn2/moments/variance

:training/SGD/gradients/bn2/moments/variance_grad/Maximum_1Maximum7training/SGD/gradients/bn2/moments/variance_grad/Prod_1<training/SGD/gradients/bn2/moments/variance_grad/Maximum_1/y*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
: 

;training/SGD/gradients/bn2/moments/variance_grad/floordiv_1FloorDiv5training/SGD/gradients/bn2/moments/variance_grad/Prod:training/SGD/gradients/bn2/moments/variance_grad/Maximum_1*
T0*'
_class
loc:@bn2/moments/variance*
_output_shapes
: 
ă
5training/SGD/gradients/bn2/moments/variance_grad/CastCast;training/SGD/gradients/bn2/moments/variance_grad/floordiv_1*

SrcT0*'
_class
loc:@bn2/moments/variance*
Truncate( *
_output_shapes
: *

DstT0

8training/SGD/gradients/bn2/moments/variance_grad/truedivRealDiv5training/SGD/gradients/bn2/moments/variance_grad/Tile5training/SGD/gradients/bn2/moments/variance_grad/Cast*
T0*'
_class
loc:@bn2/moments/variance*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
?training/SGD/gradients/bn2/moments/SquaredDifference_grad/ShapeShapefc2/BiasAdd*
_output_shapes
:*
T0*
out_type0*0
_class&
$"loc:@bn2/moments/SquaredDifference
Ä
Atraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      *0
_class&
$"loc:@bn2/moments/SquaredDifference
Ű
Otraining/SGD/gradients/bn2/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs?training/SGD/gradients/bn2/moments/SquaredDifference_grad/ShapeAtraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Shape_1*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ň
@training/SGD/gradients/bn2/moments/SquaredDifference_grad/scalarConst9^training/SGD/gradients/bn2/moments/variance_grad/truediv*
valueB
 *   @*0
_class&
$"loc:@bn2/moments/SquaredDifference*
dtype0*
_output_shapes
: 
Ľ
=training/SGD/gradients/bn2/moments/SquaredDifference_grad/MulMul@training/SGD/gradients/bn2/moments/SquaredDifference_grad/scalar8training/SGD/gradients/bn2/moments/variance_grad/truediv*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=training/SGD/gradients/bn2/moments/SquaredDifference_grad/subSubfc2/BiasAddbn2/moments/StopGradient9^training/SGD/gradients/bn2/moments/variance_grad/truediv*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
?training/SGD/gradients/bn2/moments/SquaredDifference_grad/mul_1Mul=training/SGD/gradients/bn2/moments/SquaredDifference_grad/Mul=training/SGD/gradients/bn2/moments/SquaredDifference_grad/sub*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
=training/SGD/gradients/bn2/moments/SquaredDifference_grad/SumSum?training/SGD/gradients/bn2/moments/SquaredDifference_grad/mul_1Otraining/SGD/gradients/bn2/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
ż
Atraining/SGD/gradients/bn2/moments/SquaredDifference_grad/ReshapeReshape=training/SGD/gradients/bn2/moments/SquaredDifference_grad/Sum?training/SGD/gradients/bn2/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*0
_class&
$"loc:@bn2/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
?training/SGD/gradients/bn2/moments/SquaredDifference_grad/Sum_1Sum?training/SGD/gradients/bn2/moments/SquaredDifference_grad/mul_1Qtraining/SGD/gradients/bn2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
ź
Ctraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Reshape_1Reshape?training/SGD/gradients/bn2/moments/SquaredDifference_grad/Sum_1Atraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*0
_class&
$"loc:@bn2/moments/SquaredDifference*
_output_shapes
:	
ĺ
=training/SGD/gradients/bn2/moments/SquaredDifference_grad/NegNegCtraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	*
T0*0
_class&
$"loc:@bn2/moments/SquaredDifference
˘
2training/SGD/gradients/bn2/moments/mean_grad/ShapeShapefc2/BiasAdd*
T0*
out_type0*#
_class
loc:@bn2/moments/mean*
_output_shapes
:

1training/SGD/gradients/bn2/moments/mean_grad/SizeConst*
value	B :*#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
: 
Ř
0training/SGD/gradients/bn2/moments/mean_grad/addAdd"bn2/moments/mean/reduction_indices1training/SGD/gradients/bn2/moments/mean_grad/Size*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
:
ë
0training/SGD/gradients/bn2/moments/mean_grad/modFloorMod0training/SGD/gradients/bn2/moments/mean_grad/add1training/SGD/gradients/bn2/moments/mean_grad/Size*
_output_shapes
:*
T0*#
_class
loc:@bn2/moments/mean
Ł
4training/SGD/gradients/bn2/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*#
_class
loc:@bn2/moments/mean

8training/SGD/gradients/bn2/moments/mean_grad/range/startConst*
value	B : *#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
: 

8training/SGD/gradients/bn2/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*#
_class
loc:@bn2/moments/mean
Ż
2training/SGD/gradients/bn2/moments/mean_grad/rangeRange8training/SGD/gradients/bn2/moments/mean_grad/range/start1training/SGD/gradients/bn2/moments/mean_grad/Size8training/SGD/gradients/bn2/moments/mean_grad/range/delta*#
_class
loc:@bn2/moments/mean*
_output_shapes
:*

Tidx0

7training/SGD/gradients/bn2/moments/mean_grad/Fill/valueConst*
value	B :*#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
: 

1training/SGD/gradients/bn2/moments/mean_grad/FillFill4training/SGD/gradients/bn2/moments/mean_grad/Shape_17training/SGD/gradients/bn2/moments/mean_grad/Fill/value*
T0*

index_type0*#
_class
loc:@bn2/moments/mean*
_output_shapes
:
ë
:training/SGD/gradients/bn2/moments/mean_grad/DynamicStitchDynamicStitch2training/SGD/gradients/bn2/moments/mean_grad/range0training/SGD/gradients/bn2/moments/mean_grad/mod2training/SGD/gradients/bn2/moments/mean_grad/Shape1training/SGD/gradients/bn2/moments/mean_grad/Fill*
T0*#
_class
loc:@bn2/moments/mean*
N*
_output_shapes
:

6training/SGD/gradients/bn2/moments/mean_grad/Maximum/yConst*
value	B :*#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
: 
ý
4training/SGD/gradients/bn2/moments/mean_grad/MaximumMaximum:training/SGD/gradients/bn2/moments/mean_grad/DynamicStitch6training/SGD/gradients/bn2/moments/mean_grad/Maximum/y*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
:
ő
5training/SGD/gradients/bn2/moments/mean_grad/floordivFloorDiv2training/SGD/gradients/bn2/moments/mean_grad/Shape4training/SGD/gradients/bn2/moments/mean_grad/Maximum*
_output_shapes
:*
T0*#
_class
loc:@bn2/moments/mean
˘
4training/SGD/gradients/bn2/moments/mean_grad/ReshapeReshape7training/SGD/gradients/bn2/moments/Squeeze_grad/Reshape:training/SGD/gradients/bn2/moments/mean_grad/DynamicStitch*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*#
_class
loc:@bn2/moments/mean

1training/SGD/gradients/bn2/moments/mean_grad/TileTile4training/SGD/gradients/bn2/moments/mean_grad/Reshape5training/SGD/gradients/bn2/moments/mean_grad/floordiv*

Tmultiples0*
T0*#
_class
loc:@bn2/moments/mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
¤
4training/SGD/gradients/bn2/moments/mean_grad/Shape_2Shapefc2/BiasAdd*
_output_shapes
:*
T0*
out_type0*#
_class
loc:@bn2/moments/mean
Ş
4training/SGD/gradients/bn2/moments/mean_grad/Shape_3Const*
valueB"      *#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
:
Ą
2training/SGD/gradients/bn2/moments/mean_grad/ConstConst*
valueB: *#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
:

1training/SGD/gradients/bn2/moments/mean_grad/ProdProd4training/SGD/gradients/bn2/moments/mean_grad/Shape_22training/SGD/gradients/bn2/moments/mean_grad/Const*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ł
4training/SGD/gradients/bn2/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *#
_class
loc:@bn2/moments/mean

3training/SGD/gradients/bn2/moments/mean_grad/Prod_1Prod4training/SGD/gradients/bn2/moments/mean_grad/Shape_34training/SGD/gradients/bn2/moments/mean_grad/Const_1*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 

8training/SGD/gradients/bn2/moments/mean_grad/Maximum_1/yConst*
value	B :*#
_class
loc:@bn2/moments/mean*
dtype0*
_output_shapes
: 
ö
6training/SGD/gradients/bn2/moments/mean_grad/Maximum_1Maximum3training/SGD/gradients/bn2/moments/mean_grad/Prod_18training/SGD/gradients/bn2/moments/mean_grad/Maximum_1/y*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
: 
ô
7training/SGD/gradients/bn2/moments/mean_grad/floordiv_1FloorDiv1training/SGD/gradients/bn2/moments/mean_grad/Prod6training/SGD/gradients/bn2/moments/mean_grad/Maximum_1*
T0*#
_class
loc:@bn2/moments/mean*
_output_shapes
: 
×
1training/SGD/gradients/bn2/moments/mean_grad/CastCast7training/SGD/gradients/bn2/moments/mean_grad/floordiv_1*

SrcT0*#
_class
loc:@bn2/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
ý
4training/SGD/gradients/bn2/moments/mean_grad/truedivRealDiv1training/SGD/gradients/bn2/moments/mean_grad/Tile1training/SGD/gradients/bn2/moments/mean_grad/Cast*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*#
_class
loc:@bn2/moments/mean
ú
training/SGD/gradients/AddN_4AddNEtraining/SGD/gradients/bn2/cond/batchnorm/mul_1/Switch_grad/cond_grad7training/SGD/gradients/bn2/batchnorm/mul_1_grad/ReshapeAtraining/SGD/gradients/bn2/moments/SquaredDifference_grad/Reshape4training/SGD/gradients/bn2/moments/mean_grad/truediv*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@fc2/BiasAdd
ž
3training/SGD/gradients/fc2/BiasAdd_grad/BiasAddGradBiasAddGradtraining/SGD/gradients/AddN_4*
T0*
_class
loc:@fc2/BiasAdd*
data_formatNHWC*
_output_shapes	
:
ß
-training/SGD/gradients/fc2/MatMul_grad/MatMulMatMultraining/SGD/gradients/AddN_4fc2/kernel/read*
transpose_b(*
T0*
_class
loc:@fc2/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ô
/training/SGD/gradients/fc2/MatMul_grad/MatMul_1MatMul
relu1/Relutraining/SGD/gradients/AddN_4*
T0*
_class
loc:@fc2/MatMul* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Č
/training/SGD/gradients/relu1/Relu_grad/ReluGradReluGrad-training/SGD/gradients/fc2/MatMul_grad/MatMul
relu1/Relu*
T0*
_class
loc:@relu1/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
4training/SGD/gradients/bn1/cond/Merge_grad/cond_gradSwitch/training/SGD/gradients/relu1/Relu_grad/ReluGradbn1/cond/pred_id*
T0*
_class
loc:@relu1/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ż
:training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/ShapeShapebn1/cond/batchnorm/mul_1*
T0*
out_type0*+
_class!
loc:@bn1/cond/batchnorm/add_1*
_output_shapes
:
´
<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*+
_class!
loc:@bn1/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ç
Jtraining/SGD/gradients/bn1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Shape<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Shape_1*
T0*+
_class!
loc:@bn1/cond/batchnorm/add_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
8training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/SumSum4training/SGD/gradients/bn1/cond/Merge_grad/cond_gradJtraining/SGD/gradients/bn1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@bn1/cond/batchnorm/add_1
Ť
<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/ReshapeReshape8training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Sum:training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*+
_class!
loc:@bn1/cond/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
:training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Sum_1Sum4training/SGD/gradients/bn1/cond/Merge_grad/cond_gradLtraining/SGD/gradients/bn1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@bn1/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
¤
>training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Reshape_1Reshape:training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Sum_1<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@bn1/cond/batchnorm/add_1*
_output_shapes	
:
ż
training/SGD/gradients/Switch_4Switchbn1/batchnorm/add_1bn1/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*&
_class
loc:@bn1/batchnorm/add_1
Š
!training/SGD/gradients/Identity_4Identitytraining/SGD/gradients/Switch_4*
T0*&
_class
loc:@bn1/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
training/SGD/gradients/Shape_5Shapetraining/SGD/gradients/Switch_4*
T0*
out_type0*&
_class
loc:@bn1/batchnorm/add_1*
_output_shapes
:
ľ
$training/SGD/gradients/zeros_4/ConstConst"^training/SGD/gradients/Identity_4*
valueB
 *    *&
_class
loc:@bn1/batchnorm/add_1*
dtype0*
_output_shapes
: 
Ů
training/SGD/gradients/zeros_4Filltraining/SGD/gradients/Shape_5$training/SGD/gradients/zeros_4/Const*
T0*

index_type0*&
_class
loc:@bn1/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
7training/SGD/gradients/bn1/cond/Switch_1_grad/cond_gradMergetraining/SGD/gradients/zeros_46training/SGD/gradients/bn1/cond/Merge_grad/cond_grad:1*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0*&
_class
loc:@bn1/batchnorm/add_1
Ć
:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/ShapeShapebn1/cond/batchnorm/mul_1/Switch*
T0*
out_type0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*
_output_shapes
:
´
<training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*+
_class!
loc:@bn1/cond/batchnorm/mul_1
Ç
Jtraining/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Shape<training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Shape_1*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
8training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/MulMul<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Reshapebn1/cond/batchnorm/mul*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
8training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/SumSum8training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/MulJtraining/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ť
<training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/ReshapeReshape8training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Sum:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Mul_1Mulbn1/cond/batchnorm/mul_1/Switch<training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Reshape*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Sum_1Sum:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Mul_1Ltraining/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1
¤
>training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Reshape_1Reshape:training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Sum_1<training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*
_output_shapes	
:
Î
6training/SGD/gradients/bn1/cond/batchnorm/sub_grad/NegNeg>training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Reshape_1*
T0*)
_class
loc:@bn1/cond/batchnorm/sub*
_output_shapes	
:
°
5training/SGD/gradients/bn1/batchnorm/add_1_grad/ShapeShapebn1/batchnorm/mul_1*
T0*
out_type0*&
_class
loc:@bn1/batchnorm/add_1*
_output_shapes
:
Ş
7training/SGD/gradients/bn1/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@bn1/batchnorm/add_1
ł
Etraining/SGD/gradients/bn1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/SGD/gradients/bn1/batchnorm/add_1_grad/Shape7training/SGD/gradients/bn1/batchnorm/add_1_grad/Shape_1*
T0*&
_class
loc:@bn1/batchnorm/add_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˘
3training/SGD/gradients/bn1/batchnorm/add_1_grad/SumSum7training/SGD/gradients/bn1/cond/Switch_1_grad/cond_gradEtraining/SGD/gradients/bn1/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*&
_class
loc:@bn1/batchnorm/add_1

7training/SGD/gradients/bn1/batchnorm/add_1_grad/ReshapeReshape3training/SGD/gradients/bn1/batchnorm/add_1_grad/Sum5training/SGD/gradients/bn1/batchnorm/add_1_grad/Shape*
T0*
Tshape0*&
_class
loc:@bn1/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
5training/SGD/gradients/bn1/batchnorm/add_1_grad/Sum_1Sum7training/SGD/gradients/bn1/cond/Switch_1_grad/cond_gradGtraining/SGD/gradients/bn1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*&
_class
loc:@bn1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 

9training/SGD/gradients/bn1/batchnorm/add_1_grad/Reshape_1Reshape5training/SGD/gradients/bn1/batchnorm/add_1_grad/Sum_17training/SGD/gradients/bn1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*&
_class
loc:@bn1/batchnorm/add_1*
_output_shapes	
:
Ż
training/SGD/gradients/Switch_5Switchfc1/BiasAddbn1/cond/pred_id*
T0*
_class
loc:@fc1/BiasAdd*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
!training/SGD/gradients/Identity_5Identity!training/SGD/gradients/Switch_5:1*
T0*
_class
loc:@fc1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

training/SGD/gradients/Shape_6Shape!training/SGD/gradients/Switch_5:1*
T0*
out_type0*
_class
loc:@fc1/BiasAdd*
_output_shapes
:
­
$training/SGD/gradients/zeros_5/ConstConst"^training/SGD/gradients/Identity_5*
valueB
 *    *
_class
loc:@fc1/BiasAdd*
dtype0*
_output_shapes
: 
Ń
training/SGD/gradients/zeros_5Filltraining/SGD/gradients/Shape_6$training/SGD/gradients/zeros_5/Const*
T0*

index_type0*
_class
loc:@fc1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Etraining/SGD/gradients/bn1/cond/batchnorm/mul_1/Switch_grad/cond_gradMerge<training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Reshapetraining/SGD/gradients/zeros_5*
T0*
_class
loc:@fc1/BiasAdd*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

training/SGD/gradients/Switch_6Switchbn1/beta/readbn1/cond/pred_id*
T0*
_class
loc:@bn1/beta*"
_output_shapes
::

!training/SGD/gradients/Identity_6Identity!training/SGD/gradients/Switch_6:1*
_output_shapes	
:*
T0*
_class
loc:@bn1/beta

training/SGD/gradients/Shape_7Shape!training/SGD/gradients/Switch_6:1*
_output_shapes
:*
T0*
out_type0*
_class
loc:@bn1/beta
Ş
$training/SGD/gradients/zeros_6/ConstConst"^training/SGD/gradients/Identity_6*
valueB
 *    *
_class
loc:@bn1/beta*
dtype0*
_output_shapes
: 
Á
training/SGD/gradients/zeros_6Filltraining/SGD/gradients/Shape_7$training/SGD/gradients/zeros_6/Const*
T0*

index_type0*
_class
loc:@bn1/beta*
_output_shapes	
:
ú
Ctraining/SGD/gradients/bn1/cond/batchnorm/sub/Switch_grad/cond_gradMerge>training/SGD/gradients/bn1/cond/batchnorm/add_1_grad/Reshape_1training/SGD/gradients/zeros_6*
T0*
_class
loc:@bn1/beta*
N*
_output_shapes
	:: 
â
8training/SGD/gradients/bn1/cond/batchnorm/mul_2_grad/MulMul6training/SGD/gradients/bn1/cond/batchnorm/sub_grad/Negbn1/cond/batchnorm/mul*
_output_shapes	
:*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_2
í
:training/SGD/gradients/bn1/cond/batchnorm/mul_2_grad/Mul_1Mul6training/SGD/gradients/bn1/cond/batchnorm/sub_grad/Negbn1/cond/batchnorm/mul_2/Switch*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_2*
_output_shapes	
:
¨
5training/SGD/gradients/bn1/batchnorm/mul_1_grad/ShapeShapefc1/BiasAdd*
T0*
out_type0*&
_class
loc:@bn1/batchnorm/mul_1*
_output_shapes
:
Ş
7training/SGD/gradients/bn1/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*&
_class
loc:@bn1/batchnorm/mul_1
ł
Etraining/SGD/gradients/bn1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Shape7training/SGD/gradients/bn1/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*&
_class
loc:@bn1/batchnorm/mul_1
á
3training/SGD/gradients/bn1/batchnorm/mul_1_grad/MulMul7training/SGD/gradients/bn1/batchnorm/add_1_grad/Reshapebn1/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*&
_class
loc:@bn1/batchnorm/mul_1

3training/SGD/gradients/bn1/batchnorm/mul_1_grad/SumSum3training/SGD/gradients/bn1/batchnorm/mul_1_grad/MulEtraining/SGD/gradients/bn1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*&
_class
loc:@bn1/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 

7training/SGD/gradients/bn1/batchnorm/mul_1_grad/ReshapeReshape3training/SGD/gradients/bn1/batchnorm/mul_1_grad/Sum5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*&
_class
loc:@bn1/batchnorm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Mul_1Mulfc1/BiasAdd7training/SGD/gradients/bn1/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*&
_class
loc:@bn1/batchnorm/mul_1
¤
5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Sum_1Sum5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Mul_1Gtraining/SGD/gradients/bn1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*&
_class
loc:@bn1/batchnorm/mul_1

9training/SGD/gradients/bn1/batchnorm/mul_1_grad/Reshape_1Reshape5training/SGD/gradients/bn1/batchnorm/mul_1_grad/Sum_17training/SGD/gradients/bn1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*&
_class
loc:@bn1/batchnorm/mul_1*
_output_shapes	
:
ż
1training/SGD/gradients/bn1/batchnorm/sub_grad/NegNeg9training/SGD/gradients/bn1/batchnorm/add_1_grad/Reshape_1*
T0*$
_class
loc:@bn1/batchnorm/sub*
_output_shapes	
:
ý
training/SGD/gradients/AddN_5AddN>training/SGD/gradients/bn1/cond/batchnorm/mul_1_grad/Reshape_1:training/SGD/gradients/bn1/cond/batchnorm/mul_2_grad/Mul_1*
T0*+
_class!
loc:@bn1/cond/batchnorm/mul_1*
N*
_output_shapes	
:
Ě
6training/SGD/gradients/bn1/cond/batchnorm/mul_grad/MulMultraining/SGD/gradients/AddN_5bn1/cond/batchnorm/mul/Switch*
T0*)
_class
loc:@bn1/cond/batchnorm/mul*
_output_shapes	
:
É
8training/SGD/gradients/bn1/cond/batchnorm/mul_grad/Mul_1Multraining/SGD/gradients/AddN_5bn1/cond/batchnorm/Rsqrt*
T0*)
_class
loc:@bn1/cond/batchnorm/mul*
_output_shapes	
:
ń
training/SGD/gradients/AddN_6AddNCtraining/SGD/gradients/bn1/cond/batchnorm/sub/Switch_grad/cond_grad9training/SGD/gradients/bn1/batchnorm/add_1_grad/Reshape_1*
T0*
_class
loc:@bn1/beta*
N*
_output_shapes	
:
Î
3training/SGD/gradients/bn1/batchnorm/mul_2_grad/MulMul1training/SGD/gradients/bn1/batchnorm/sub_grad/Negbn1/batchnorm/mul*
T0*&
_class
loc:@bn1/batchnorm/mul_2*
_output_shapes	
:
Ň
5training/SGD/gradients/bn1/batchnorm/mul_2_grad/Mul_1Mul1training/SGD/gradients/bn1/batchnorm/sub_grad/Negbn1/moments/Squeeze*
T0*&
_class
loc:@bn1/batchnorm/mul_2*
_output_shapes	
:

training/SGD/gradients/Switch_7Switchbn1/gamma/readbn1/cond/pred_id*"
_output_shapes
::*
T0*
_class
loc:@bn1/gamma

!training/SGD/gradients/Identity_7Identity!training/SGD/gradients/Switch_7:1*
_output_shapes	
:*
T0*
_class
loc:@bn1/gamma

training/SGD/gradients/Shape_8Shape!training/SGD/gradients/Switch_7:1*
T0*
out_type0*
_class
loc:@bn1/gamma*
_output_shapes
:
Ť
$training/SGD/gradients/zeros_7/ConstConst"^training/SGD/gradients/Identity_7*
valueB
 *    *
_class
loc:@bn1/gamma*
dtype0*
_output_shapes
: 
Â
training/SGD/gradients/zeros_7Filltraining/SGD/gradients/Shape_8$training/SGD/gradients/zeros_7/Const*
T0*

index_type0*
_class
loc:@bn1/gamma*
_output_shapes	
:
ő
Ctraining/SGD/gradients/bn1/cond/batchnorm/mul/Switch_grad/cond_gradMerge8training/SGD/gradients/bn1/cond/batchnorm/mul_grad/Mul_1training/SGD/gradients/zeros_7*
T0*
_class
loc:@bn1/gamma*
N*
_output_shapes
	:: 
Ž
5training/SGD/gradients/bn1/moments/Squeeze_grad/ShapeConst*
valueB"      *&
_class
loc:@bn1/moments/Squeeze*
dtype0*
_output_shapes
:

7training/SGD/gradients/bn1/moments/Squeeze_grad/ReshapeReshape3training/SGD/gradients/bn1/batchnorm/mul_2_grad/Mul5training/SGD/gradients/bn1/moments/Squeeze_grad/Shape*
T0*
Tshape0*&
_class
loc:@bn1/moments/Squeeze*
_output_shapes
:	
î
training/SGD/gradients/AddN_7AddN9training/SGD/gradients/bn1/batchnorm/mul_1_grad/Reshape_15training/SGD/gradients/bn1/batchnorm/mul_2_grad/Mul_1*
T0*&
_class
loc:@bn1/batchnorm/mul_1*
N*
_output_shapes	
:
ł
1training/SGD/gradients/bn1/batchnorm/mul_grad/MulMultraining/SGD/gradients/AddN_7bn1/gamma/read*
T0*$
_class
loc:@bn1/batchnorm/mul*
_output_shapes	
:
ş
3training/SGD/gradients/bn1/batchnorm/mul_grad/Mul_1Multraining/SGD/gradients/AddN_7bn1/batchnorm/Rsqrt*
T0*$
_class
loc:@bn1/batchnorm/mul*
_output_shapes	
:
Ü
9training/SGD/gradients/bn1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGradbn1/batchnorm/Rsqrt1training/SGD/gradients/bn1/batchnorm/mul_grad/Mul*
T0*&
_class
loc:@bn1/batchnorm/Rsqrt*
_output_shapes	
:
ě
training/SGD/gradients/AddN_8AddNCtraining/SGD/gradients/bn1/cond/batchnorm/mul/Switch_grad/cond_grad3training/SGD/gradients/bn1/batchnorm/mul_grad/Mul_1*
T0*
_class
loc:@bn1/gamma*
N*
_output_shapes	
:
¤
3training/SGD/gradients/bn1/batchnorm/add_grad/ShapeConst*
valueB:*$
_class
loc:@bn1/batchnorm/add*
dtype0*
_output_shapes
:

5training/SGD/gradients/bn1/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *$
_class
loc:@bn1/batchnorm/add
Ť
Ctraining/SGD/gradients/bn1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs3training/SGD/gradients/bn1/batchnorm/add_grad/Shape5training/SGD/gradients/bn1/batchnorm/add_grad/Shape_1*
T0*$
_class
loc:@bn1/batchnorm/add*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
1training/SGD/gradients/bn1/batchnorm/add_grad/SumSum9training/SGD/gradients/bn1/batchnorm/Rsqrt_grad/RsqrtGradCtraining/SGD/gradients/bn1/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes	
:*

Tidx0*
	keep_dims( *
T0*$
_class
loc:@bn1/batchnorm/add

5training/SGD/gradients/bn1/batchnorm/add_grad/ReshapeReshape1training/SGD/gradients/bn1/batchnorm/add_grad/Sum3training/SGD/gradients/bn1/batchnorm/add_grad/Shape*
T0*
Tshape0*$
_class
loc:@bn1/batchnorm/add*
_output_shapes	
:
 
3training/SGD/gradients/bn1/batchnorm/add_grad/Sum_1Sum9training/SGD/gradients/bn1/batchnorm/Rsqrt_grad/RsqrtGradEtraining/SGD/gradients/bn1/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*$
_class
loc:@bn1/batchnorm/add*
_output_shapes
: 

7training/SGD/gradients/bn1/batchnorm/add_grad/Reshape_1Reshape3training/SGD/gradients/bn1/batchnorm/add_grad/Sum_15training/SGD/gradients/bn1/batchnorm/add_grad/Shape_1*
T0*
Tshape0*$
_class
loc:@bn1/batchnorm/add*
_output_shapes
: 
˛
7training/SGD/gradients/bn1/moments/Squeeze_1_grad/ShapeConst*
valueB"      *(
_class
loc:@bn1/moments/Squeeze_1*
dtype0*
_output_shapes
:

9training/SGD/gradients/bn1/moments/Squeeze_1_grad/ReshapeReshape5training/SGD/gradients/bn1/batchnorm/add_grad/Reshape7training/SGD/gradients/bn1/moments/Squeeze_1_grad/Shape*
_output_shapes
:	*
T0*
Tshape0*(
_class
loc:@bn1/moments/Squeeze_1
ź
6training/SGD/gradients/bn1/moments/variance_grad/ShapeShapebn1/moments/SquaredDifference*
T0*
out_type0*'
_class
loc:@bn1/moments/variance*
_output_shapes
:
 
5training/SGD/gradients/bn1/moments/variance_grad/SizeConst*
value	B :*'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
: 
č
4training/SGD/gradients/bn1/moments/variance_grad/addAdd&bn1/moments/variance/reduction_indices5training/SGD/gradients/bn1/moments/variance_grad/Size*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
:
ű
4training/SGD/gradients/bn1/moments/variance_grad/modFloorMod4training/SGD/gradients/bn1/moments/variance_grad/add5training/SGD/gradients/bn1/moments/variance_grad/Size*
_output_shapes
:*
T0*'
_class
loc:@bn1/moments/variance
Ť
8training/SGD/gradients/bn1/moments/variance_grad/Shape_1Const*
valueB:*'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
:
§
<training/SGD/gradients/bn1/moments/variance_grad/range/startConst*
value	B : *'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
: 
§
<training/SGD/gradients/bn1/moments/variance_grad/range/deltaConst*
value	B :*'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
: 
Ă
6training/SGD/gradients/bn1/moments/variance_grad/rangeRange<training/SGD/gradients/bn1/moments/variance_grad/range/start5training/SGD/gradients/bn1/moments/variance_grad/Size<training/SGD/gradients/bn1/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*'
_class
loc:@bn1/moments/variance
Ś
;training/SGD/gradients/bn1/moments/variance_grad/Fill/valueConst*
value	B :*'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
: 

5training/SGD/gradients/bn1/moments/variance_grad/FillFill8training/SGD/gradients/bn1/moments/variance_grad/Shape_1;training/SGD/gradients/bn1/moments/variance_grad/Fill/value*
T0*

index_type0*'
_class
loc:@bn1/moments/variance*
_output_shapes
:

>training/SGD/gradients/bn1/moments/variance_grad/DynamicStitchDynamicStitch6training/SGD/gradients/bn1/moments/variance_grad/range4training/SGD/gradients/bn1/moments/variance_grad/mod6training/SGD/gradients/bn1/moments/variance_grad/Shape5training/SGD/gradients/bn1/moments/variance_grad/Fill*
T0*'
_class
loc:@bn1/moments/variance*
N*
_output_shapes
:
Ľ
:training/SGD/gradients/bn1/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@bn1/moments/variance

8training/SGD/gradients/bn1/moments/variance_grad/MaximumMaximum>training/SGD/gradients/bn1/moments/variance_grad/DynamicStitch:training/SGD/gradients/bn1/moments/variance_grad/Maximum/y*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
:

9training/SGD/gradients/bn1/moments/variance_grad/floordivFloorDiv6training/SGD/gradients/bn1/moments/variance_grad/Shape8training/SGD/gradients/bn1/moments/variance_grad/Maximum*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
:
°
8training/SGD/gradients/bn1/moments/variance_grad/ReshapeReshape9training/SGD/gradients/bn1/moments/Squeeze_1_grad/Reshape>training/SGD/gradients/bn1/moments/variance_grad/DynamicStitch*
T0*
Tshape0*'
_class
loc:@bn1/moments/variance*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
¨
5training/SGD/gradients/bn1/moments/variance_grad/TileTile8training/SGD/gradients/bn1/moments/variance_grad/Reshape9training/SGD/gradients/bn1/moments/variance_grad/floordiv*
T0*'
_class
loc:@bn1/moments/variance*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0
ž
8training/SGD/gradients/bn1/moments/variance_grad/Shape_2Shapebn1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@bn1/moments/variance
˛
8training/SGD/gradients/bn1/moments/variance_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB"      *'
_class
loc:@bn1/moments/variance
Š
6training/SGD/gradients/bn1/moments/variance_grad/ConstConst*
valueB: *'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
:

5training/SGD/gradients/bn1/moments/variance_grad/ProdProd8training/SGD/gradients/bn1/moments/variance_grad/Shape_26training/SGD/gradients/bn1/moments/variance_grad/Const*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ť
8training/SGD/gradients/bn1/moments/variance_grad/Const_1Const*
valueB: *'
_class
loc:@bn1/moments/variance*
dtype0*
_output_shapes
:

7training/SGD/gradients/bn1/moments/variance_grad/Prod_1Prod8training/SGD/gradients/bn1/moments/variance_grad/Shape_38training/SGD/gradients/bn1/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
: 
§
<training/SGD/gradients/bn1/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@bn1/moments/variance

:training/SGD/gradients/bn1/moments/variance_grad/Maximum_1Maximum7training/SGD/gradients/bn1/moments/variance_grad/Prod_1<training/SGD/gradients/bn1/moments/variance_grad/Maximum_1/y*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
: 

;training/SGD/gradients/bn1/moments/variance_grad/floordiv_1FloorDiv5training/SGD/gradients/bn1/moments/variance_grad/Prod:training/SGD/gradients/bn1/moments/variance_grad/Maximum_1*
T0*'
_class
loc:@bn1/moments/variance*
_output_shapes
: 
ă
5training/SGD/gradients/bn1/moments/variance_grad/CastCast;training/SGD/gradients/bn1/moments/variance_grad/floordiv_1*

SrcT0*'
_class
loc:@bn1/moments/variance*
Truncate( *
_output_shapes
: *

DstT0

8training/SGD/gradients/bn1/moments/variance_grad/truedivRealDiv5training/SGD/gradients/bn1/moments/variance_grad/Tile5training/SGD/gradients/bn1/moments/variance_grad/Cast*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*'
_class
loc:@bn1/moments/variance
ź
?training/SGD/gradients/bn1/moments/SquaredDifference_grad/ShapeShapefc1/BiasAdd*
T0*
out_type0*0
_class&
$"loc:@bn1/moments/SquaredDifference*
_output_shapes
:
Ä
Atraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Shape_1Const*
valueB"      *0
_class&
$"loc:@bn1/moments/SquaredDifference*
dtype0*
_output_shapes
:
Ű
Otraining/SGD/gradients/bn1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs?training/SGD/gradients/bn1/moments/SquaredDifference_grad/ShapeAtraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Shape_1*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ň
@training/SGD/gradients/bn1/moments/SquaredDifference_grad/scalarConst9^training/SGD/gradients/bn1/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @*0
_class&
$"loc:@bn1/moments/SquaredDifference
Ľ
=training/SGD/gradients/bn1/moments/SquaredDifference_grad/MulMul@training/SGD/gradients/bn1/moments/SquaredDifference_grad/scalar8training/SGD/gradients/bn1/moments/variance_grad/truediv*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference

=training/SGD/gradients/bn1/moments/SquaredDifference_grad/subSubfc1/BiasAddbn1/moments/StopGradient9^training/SGD/gradients/bn1/moments/variance_grad/truediv*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
?training/SGD/gradients/bn1/moments/SquaredDifference_grad/mul_1Mul=training/SGD/gradients/bn1/moments/SquaredDifference_grad/Mul=training/SGD/gradients/bn1/moments/SquaredDifference_grad/sub*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
=training/SGD/gradients/bn1/moments/SquaredDifference_grad/SumSum?training/SGD/gradients/bn1/moments/SquaredDifference_grad/mul_1Otraining/SGD/gradients/bn1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
ż
Atraining/SGD/gradients/bn1/moments/SquaredDifference_grad/ReshapeReshape=training/SGD/gradients/bn1/moments/SquaredDifference_grad/Sum?training/SGD/gradients/bn1/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*0
_class&
$"loc:@bn1/moments/SquaredDifference*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
?training/SGD/gradients/bn1/moments/SquaredDifference_grad/Sum_1Sum?training/SGD/gradients/bn1/moments/SquaredDifference_grad/mul_1Qtraining/SGD/gradients/bn1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
ź
Ctraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Reshape_1Reshape?training/SGD/gradients/bn1/moments/SquaredDifference_grad/Sum_1Atraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*0
_class&
$"loc:@bn1/moments/SquaredDifference*
_output_shapes
:	
ĺ
=training/SGD/gradients/bn1/moments/SquaredDifference_grad/NegNegCtraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	*
T0*0
_class&
$"loc:@bn1/moments/SquaredDifference
˘
2training/SGD/gradients/bn1/moments/mean_grad/ShapeShapefc1/BiasAdd*
T0*
out_type0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:

1training/SGD/gradients/bn1/moments/mean_grad/SizeConst*
value	B :*#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
: 
Ř
0training/SGD/gradients/bn1/moments/mean_grad/addAdd"bn1/moments/mean/reduction_indices1training/SGD/gradients/bn1/moments/mean_grad/Size*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:
ë
0training/SGD/gradients/bn1/moments/mean_grad/modFloorMod0training/SGD/gradients/bn1/moments/mean_grad/add1training/SGD/gradients/bn1/moments/mean_grad/Size*
_output_shapes
:*
T0*#
_class
loc:@bn1/moments/mean
Ł
4training/SGD/gradients/bn1/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*#
_class
loc:@bn1/moments/mean

8training/SGD/gradients/bn1/moments/mean_grad/range/startConst*
value	B : *#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
: 

8training/SGD/gradients/bn1/moments/mean_grad/range/deltaConst*
value	B :*#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
: 
Ż
2training/SGD/gradients/bn1/moments/mean_grad/rangeRange8training/SGD/gradients/bn1/moments/mean_grad/range/start1training/SGD/gradients/bn1/moments/mean_grad/Size8training/SGD/gradients/bn1/moments/mean_grad/range/delta*#
_class
loc:@bn1/moments/mean*
_output_shapes
:*

Tidx0

7training/SGD/gradients/bn1/moments/mean_grad/Fill/valueConst*
value	B :*#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
: 

1training/SGD/gradients/bn1/moments/mean_grad/FillFill4training/SGD/gradients/bn1/moments/mean_grad/Shape_17training/SGD/gradients/bn1/moments/mean_grad/Fill/value*
T0*

index_type0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:
ë
:training/SGD/gradients/bn1/moments/mean_grad/DynamicStitchDynamicStitch2training/SGD/gradients/bn1/moments/mean_grad/range0training/SGD/gradients/bn1/moments/mean_grad/mod2training/SGD/gradients/bn1/moments/mean_grad/Shape1training/SGD/gradients/bn1/moments/mean_grad/Fill*
N*
_output_shapes
:*
T0*#
_class
loc:@bn1/moments/mean

6training/SGD/gradients/bn1/moments/mean_grad/Maximum/yConst*
value	B :*#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
: 
ý
4training/SGD/gradients/bn1/moments/mean_grad/MaximumMaximum:training/SGD/gradients/bn1/moments/mean_grad/DynamicStitch6training/SGD/gradients/bn1/moments/mean_grad/Maximum/y*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:
ő
5training/SGD/gradients/bn1/moments/mean_grad/floordivFloorDiv2training/SGD/gradients/bn1/moments/mean_grad/Shape4training/SGD/gradients/bn1/moments/mean_grad/Maximum*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:
˘
4training/SGD/gradients/bn1/moments/mean_grad/ReshapeReshape7training/SGD/gradients/bn1/moments/Squeeze_grad/Reshape:training/SGD/gradients/bn1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*#
_class
loc:@bn1/moments/mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

1training/SGD/gradients/bn1/moments/mean_grad/TileTile4training/SGD/gradients/bn1/moments/mean_grad/Reshape5training/SGD/gradients/bn1/moments/mean_grad/floordiv*
T0*#
_class
loc:@bn1/moments/mean*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0
¤
4training/SGD/gradients/bn1/moments/mean_grad/Shape_2Shapefc1/BiasAdd*
T0*
out_type0*#
_class
loc:@bn1/moments/mean*
_output_shapes
:
Ş
4training/SGD/gradients/bn1/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB"      *#
_class
loc:@bn1/moments/mean
Ą
2training/SGD/gradients/bn1/moments/mean_grad/ConstConst*
valueB: *#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
:

1training/SGD/gradients/bn1/moments/mean_grad/ProdProd4training/SGD/gradients/bn1/moments/mean_grad/Shape_22training/SGD/gradients/bn1/moments/mean_grad/Const*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ł
4training/SGD/gradients/bn1/moments/mean_grad/Const_1Const*
valueB: *#
_class
loc:@bn1/moments/mean*
dtype0*
_output_shapes
:

3training/SGD/gradients/bn1/moments/mean_grad/Prod_1Prod4training/SGD/gradients/bn1/moments/mean_grad/Shape_34training/SGD/gradients/bn1/moments/mean_grad/Const_1*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 

8training/SGD/gradients/bn1/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*#
_class
loc:@bn1/moments/mean
ö
6training/SGD/gradients/bn1/moments/mean_grad/Maximum_1Maximum3training/SGD/gradients/bn1/moments/mean_grad/Prod_18training/SGD/gradients/bn1/moments/mean_grad/Maximum_1/y*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
: 
ô
7training/SGD/gradients/bn1/moments/mean_grad/floordiv_1FloorDiv1training/SGD/gradients/bn1/moments/mean_grad/Prod6training/SGD/gradients/bn1/moments/mean_grad/Maximum_1*
T0*#
_class
loc:@bn1/moments/mean*
_output_shapes
: 
×
1training/SGD/gradients/bn1/moments/mean_grad/CastCast7training/SGD/gradients/bn1/moments/mean_grad/floordiv_1*

SrcT0*#
_class
loc:@bn1/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
ý
4training/SGD/gradients/bn1/moments/mean_grad/truedivRealDiv1training/SGD/gradients/bn1/moments/mean_grad/Tile1training/SGD/gradients/bn1/moments/mean_grad/Cast*
T0*#
_class
loc:@bn1/moments/mean*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
training/SGD/gradients/AddN_9AddNEtraining/SGD/gradients/bn1/cond/batchnorm/mul_1/Switch_grad/cond_grad7training/SGD/gradients/bn1/batchnorm/mul_1_grad/ReshapeAtraining/SGD/gradients/bn1/moments/SquaredDifference_grad/Reshape4training/SGD/gradients/bn1/moments/mean_grad/truediv*
T0*
_class
loc:@fc1/BiasAdd*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
3training/SGD/gradients/fc1/BiasAdd_grad/BiasAddGradBiasAddGradtraining/SGD/gradients/AddN_9*
T0*
_class
loc:@fc1/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ţ
-training/SGD/gradients/fc1/MatMul_grad/MatMulMatMultraining/SGD/gradients/AddN_9fc1/kernel/read*
T0*
_class
loc:@fc1/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
transpose_a( *
transpose_b(
Ő
/training/SGD/gradients/fc1/MatMul_grad/MatMul_1MatMulinput_tensortraining/SGD/gradients/AddN_9*
_output_shapes
:	/*
transpose_a(*
transpose_b( *
T0*
_class
loc:@fc1/MatMul
^
training/SGD/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
¨
training/SGD/AssignAdd	AssignAddSGD/iterationstraining/SGD/AssignAdd/value*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: *
use_locking( 
s
"training/SGD/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"/      
]
training/SGD/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/SGD/zerosFill"training/SGD/zeros/shape_as_tensortraining/SGD/zeros/Const*
_output_shapes
:	/*
T0*

index_type0

training/SGD/Variable
VariableV2*
dtype0*
_output_shapes
:	/*
	container *
shape:	/*
shared_name 
Î
training/SGD/Variable/AssignAssigntraining/SGD/Variabletraining/SGD/zeros*
use_locking(*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes
:	/

training/SGD/Variable/readIdentitytraining/SGD/Variable*
T0*(
_class
loc:@training/SGD/Variable*
_output_shapes
:	/
c
training/SGD/zeros_1Const*
dtype0*
_output_shapes	
:*
valueB*    

training/SGD/Variable_1
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ň
training/SGD/Variable_1/AssignAssigntraining/SGD/Variable_1training/SGD/zeros_1*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1

training/SGD/Variable_1/readIdentitytraining/SGD/Variable_1*
T0**
_class 
loc:@training/SGD/Variable_1*
_output_shapes	
:
c
training/SGD/zeros_2Const*
dtype0*
_output_shapes	
:*
valueB*    

training/SGD/Variable_2
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ň
training/SGD/Variable_2/AssignAssigntraining/SGD/Variable_2training/SGD/zeros_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*
_output_shapes	
:

training/SGD/Variable_2/readIdentitytraining/SGD/Variable_2*
T0**
_class 
loc:@training/SGD/Variable_2*
_output_shapes	
:
c
training/SGD/zeros_3Const*
valueB*    *
dtype0*
_output_shapes	
:

training/SGD/Variable_3
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ň
training/SGD/Variable_3/AssignAssigntraining/SGD/Variable_3training/SGD/zeros_3*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3

training/SGD/Variable_3/readIdentitytraining/SGD/Variable_3*
_output_shapes	
:*
T0**
_class 
loc:@training/SGD/Variable_3
u
$training/SGD/zeros_4/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
_
training/SGD/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/SGD/zeros_4Fill$training/SGD/zeros_4/shape_as_tensortraining/SGD/zeros_4/Const*
T0*

index_type0* 
_output_shapes
:


training/SGD/Variable_4
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
×
training/SGD/Variable_4/AssignAssigntraining/SGD/Variable_4training/SGD/zeros_4*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(* 
_output_shapes
:


training/SGD/Variable_4/readIdentitytraining/SGD/Variable_4*
T0**
_class 
loc:@training/SGD/Variable_4* 
_output_shapes
:

c
training/SGD/zeros_5Const*
dtype0*
_output_shapes	
:*
valueB*    

training/SGD/Variable_5
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ň
training/SGD/Variable_5/AssignAssigntraining/SGD/Variable_5training/SGD/zeros_5*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes	
:

training/SGD/Variable_5/readIdentitytraining/SGD/Variable_5*
T0**
_class 
loc:@training/SGD/Variable_5*
_output_shapes	
:
c
training/SGD/zeros_6Const*
dtype0*
_output_shapes	
:*
valueB*    

training/SGD/Variable_6
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ň
training/SGD/Variable_6/AssignAssigntraining/SGD/Variable_6training/SGD/zeros_6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_6

training/SGD/Variable_6/readIdentitytraining/SGD/Variable_6*
_output_shapes	
:*
T0**
_class 
loc:@training/SGD/Variable_6
c
training/SGD/zeros_7Const*
dtype0*
_output_shapes	
:*
valueB*    

training/SGD/Variable_7
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ň
training/SGD/Variable_7/AssignAssigntraining/SGD/Variable_7training/SGD/zeros_7*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_7*
validate_shape(*
_output_shapes	
:

training/SGD/Variable_7/readIdentitytraining/SGD/Variable_7*
T0**
_class 
loc:@training/SGD/Variable_7*
_output_shapes	
:
k
training/SGD/zeros_8Const*
dtype0*
_output_shapes
:	*
valueB	*    

training/SGD/Variable_8
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
Ö
training/SGD/Variable_8/AssignAssigntraining/SGD/Variable_8training/SGD/zeros_8*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_8*
validate_shape(*
_output_shapes
:	

training/SGD/Variable_8/readIdentitytraining/SGD/Variable_8*
_output_shapes
:	*
T0**
_class 
loc:@training/SGD/Variable_8
a
training/SGD/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:

training/SGD/Variable_9
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ń
training/SGD/Variable_9/AssignAssigntraining/SGD/Variable_9training/SGD/zeros_9*
T0**
_class 
loc:@training/SGD/Variable_9*
validate_shape(*
_output_shapes
:*
use_locking(

training/SGD/Variable_9/readIdentitytraining/SGD/Variable_9*
T0**
_class 
loc:@training/SGD/Variable_9*
_output_shapes
:
p
training/SGD/mulMulSGD/momentum/readtraining/SGD/Variable/read*
T0*
_output_shapes
:	/

training/SGD/mul_1MulSGD/lr/read/training/SGD/gradients/fc1/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	/
g
training/SGD/subSubtraining/SGD/multraining/SGD/mul_1*
T0*
_output_shapes
:	/
Ă
training/SGD/AssignAssigntraining/SGD/Variabletraining/SGD/sub*
use_locking(*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes
:	/
d
training/SGD/addAddfc1/kernel/readtraining/SGD/sub*
T0*
_output_shapes
:	/
Ż
training/SGD/Assign_1Assign
fc1/kerneltraining/SGD/add*
T0*
_class
loc:@fc1/kernel*
validate_shape(*
_output_shapes
:	/*
use_locking(
p
training/SGD/mul_2MulSGD/momentum/readtraining/SGD/Variable_1/read*
T0*
_output_shapes	
:

training/SGD/mul_3MulSGD/lr/read3training/SGD/gradients/fc1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
g
training/SGD/sub_1Subtraining/SGD/mul_2training/SGD/mul_3*
_output_shapes	
:*
T0
Ç
training/SGD/Assign_2Assigntraining/SGD/Variable_1training/SGD/sub_1*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes	
:
b
training/SGD/add_1Addfc1/bias/readtraining/SGD/sub_1*
T0*
_output_shapes	
:
Š
training/SGD/Assign_3Assignfc1/biastraining/SGD/add_1*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
p
training/SGD/mul_4MulSGD/momentum/readtraining/SGD/Variable_2/read*
T0*
_output_shapes	
:
k
training/SGD/mul_5MulSGD/lr/readtraining/SGD/gradients/AddN_8*
_output_shapes	
:*
T0
g
training/SGD/sub_2Subtraining/SGD/mul_4training/SGD/mul_5*
_output_shapes	
:*
T0
Ç
training/SGD/Assign_4Assigntraining/SGD/Variable_2training/SGD/sub_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*
_output_shapes	
:
c
training/SGD/add_2Addbn1/gamma/readtraining/SGD/sub_2*
T0*
_output_shapes	
:
Ť
training/SGD/Assign_5Assign	bn1/gammatraining/SGD/add_2*
T0*
_class
loc:@bn1/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
p
training/SGD/mul_6MulSGD/momentum/readtraining/SGD/Variable_3/read*
_output_shapes	
:*
T0
k
training/SGD/mul_7MulSGD/lr/readtraining/SGD/gradients/AddN_6*
T0*
_output_shapes	
:
g
training/SGD/sub_3Subtraining/SGD/mul_6training/SGD/mul_7*
T0*
_output_shapes	
:
Ç
training/SGD/Assign_6Assigntraining/SGD/Variable_3training/SGD/sub_3*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes	
:*
use_locking(
b
training/SGD/add_3Addbn1/beta/readtraining/SGD/sub_3*
T0*
_output_shapes	
:
Š
training/SGD/Assign_7Assignbn1/betatraining/SGD/add_3*
use_locking(*
T0*
_class
loc:@bn1/beta*
validate_shape(*
_output_shapes	
:
u
training/SGD/mul_8MulSGD/momentum/readtraining/SGD/Variable_4/read*
T0* 
_output_shapes
:


training/SGD/mul_9MulSGD/lr/read/training/SGD/gradients/fc2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

l
training/SGD/sub_4Subtraining/SGD/mul_8training/SGD/mul_9*
T0* 
_output_shapes
:

Ě
training/SGD/Assign_8Assigntraining/SGD/Variable_4training/SGD/sub_4*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(* 
_output_shapes
:
*
use_locking(
i
training/SGD/add_4Addfc2/kernel/readtraining/SGD/sub_4*
T0* 
_output_shapes
:

˛
training/SGD/Assign_9Assign
fc2/kerneltraining/SGD/add_4*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(* 
_output_shapes
:

q
training/SGD/mul_10MulSGD/momentum/readtraining/SGD/Variable_5/read*
T0*
_output_shapes	
:

training/SGD/mul_11MulSGD/lr/read3training/SGD/gradients/fc2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
i
training/SGD/sub_5Subtraining/SGD/mul_10training/SGD/mul_11*
_output_shapes	
:*
T0
Č
training/SGD/Assign_10Assigntraining/SGD/Variable_5training/SGD/sub_5*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes	
:
b
training/SGD/add_5Addfc2/bias/readtraining/SGD/sub_5*
T0*
_output_shapes	
:
Ş
training/SGD/Assign_11Assignfc2/biastraining/SGD/add_5*
use_locking(*
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes	
:
q
training/SGD/mul_12MulSGD/momentum/readtraining/SGD/Variable_6/read*
T0*
_output_shapes	
:
l
training/SGD/mul_13MulSGD/lr/readtraining/SGD/gradients/AddN_3*
T0*
_output_shapes	
:
i
training/SGD/sub_6Subtraining/SGD/mul_12training/SGD/mul_13*
T0*
_output_shapes	
:
Č
training/SGD/Assign_12Assigntraining/SGD/Variable_6training/SGD/sub_6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_6
c
training/SGD/add_6Addbn2/gamma/readtraining/SGD/sub_6*
T0*
_output_shapes	
:
Ź
training/SGD/Assign_13Assign	bn2/gammatraining/SGD/add_6*
use_locking(*
T0*
_class
loc:@bn2/gamma*
validate_shape(*
_output_shapes	
:
q
training/SGD/mul_14MulSGD/momentum/readtraining/SGD/Variable_7/read*
T0*
_output_shapes	
:
l
training/SGD/mul_15MulSGD/lr/readtraining/SGD/gradients/AddN_1*
_output_shapes	
:*
T0
i
training/SGD/sub_7Subtraining/SGD/mul_14training/SGD/mul_15*
T0*
_output_shapes	
:
Č
training/SGD/Assign_14Assigntraining/SGD/Variable_7training/SGD/sub_7*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_7*
validate_shape(*
_output_shapes	
:
b
training/SGD/add_7Addbn2/beta/readtraining/SGD/sub_7*
T0*
_output_shapes	
:
Ş
training/SGD/Assign_15Assignbn2/betatraining/SGD/add_7*
use_locking(*
T0*
_class
loc:@bn2/beta*
validate_shape(*
_output_shapes	
:
u
training/SGD/mul_16MulSGD/momentum/readtraining/SGD/Variable_8/read*
T0*
_output_shapes
:	

training/SGD/mul_17MulSGD/lr/read2training/SGD/gradients/output/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
m
training/SGD/sub_8Subtraining/SGD/mul_16training/SGD/mul_17*
T0*
_output_shapes
:	
Ě
training/SGD/Assign_16Assigntraining/SGD/Variable_8training/SGD/sub_8*
T0**
_class 
loc:@training/SGD/Variable_8*
validate_shape(*
_output_shapes
:	*
use_locking(
k
training/SGD/add_8Addoutput/kernel/readtraining/SGD/sub_8*
T0*
_output_shapes
:	
¸
training/SGD/Assign_17Assignoutput/kerneltraining/SGD/add_8*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes
:	
p
training/SGD/mul_18MulSGD/momentum/readtraining/SGD/Variable_9/read*
_output_shapes
:*
T0

training/SGD/mul_19MulSGD/lr/read6training/SGD/gradients/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
h
training/SGD/sub_9Subtraining/SGD/mul_18training/SGD/mul_19*
_output_shapes
:*
T0
Ç
training/SGD/Assign_18Assigntraining/SGD/Variable_9training/SGD/sub_9*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_9*
validate_shape(*
_output_shapes
:
d
training/SGD/add_9Addoutput/bias/readtraining/SGD/sub_9*
T0*
_output_shapes
:
Ż
training/SGD/Assign_19Assignoutput/biastraining/SGD/add_9*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:

training/group_depsNoOp^bn1/AssignMovingAvg^bn1/AssignMovingAvg_1^bn2/AssignMovingAvg^bn2/AssignMovingAvg_1	^loss/mul^metrics/rmse/Mean_1^training/SGD/Assign^training/SGD/AssignAdd^training/SGD/Assign_1^training/SGD/Assign_10^training/SGD/Assign_11^training/SGD/Assign_12^training/SGD/Assign_13^training/SGD/Assign_14^training/SGD/Assign_15^training/SGD/Assign_16^training/SGD/Assign_17^training/SGD/Assign_18^training/SGD/Assign_19^training/SGD/Assign_2^training/SGD/Assign_3^training/SGD/Assign_4^training/SGD/Assign_5^training/SGD/Assign_6^training/SGD/Assign_7^training/SGD/Assign_8^training/SGD/Assign_9
3

group_depsNoOp	^loss/mul^metrics/rmse/Mean_1
~
IsVariableInitializedIsVariableInitialized
fc1/kernel*
_class
loc:@fc1/kernel*
dtype0*
_output_shapes
: 
|
IsVariableInitialized_1IsVariableInitializedfc1/bias*
_class
loc:@fc1/bias*
dtype0*
_output_shapes
: 
~
IsVariableInitialized_2IsVariableInitialized	bn1/gamma*
dtype0*
_output_shapes
: *
_class
loc:@bn1/gamma
|
IsVariableInitialized_3IsVariableInitializedbn1/beta*
dtype0*
_output_shapes
: *
_class
loc:@bn1/beta

IsVariableInitialized_4IsVariableInitializedbn1/moving_mean*
dtype0*
_output_shapes
: *"
_class
loc:@bn1/moving_mean

IsVariableInitialized_5IsVariableInitializedbn1/moving_variance*
dtype0*
_output_shapes
: *&
_class
loc:@bn1/moving_variance

IsVariableInitialized_6IsVariableInitializedbn1/moving_mean/biased*
dtype0*
_output_shapes
: *"
_class
loc:@bn1/moving_mean

IsVariableInitialized_7IsVariableInitializedbn1/moving_mean/local_step*"
_class
loc:@bn1/moving_mean*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedbn1/moving_variance/biased*&
_class
loc:@bn1/moving_variance*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedbn1/moving_variance/local_step*&
_class
loc:@bn1/moving_variance*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitialized
fc2/kernel*
_class
loc:@fc2/kernel*
dtype0*
_output_shapes
: 
}
IsVariableInitialized_11IsVariableInitializedfc2/bias*
dtype0*
_output_shapes
: *
_class
loc:@fc2/bias

IsVariableInitialized_12IsVariableInitialized	bn2/gamma*
_class
loc:@bn2/gamma*
dtype0*
_output_shapes
: 
}
IsVariableInitialized_13IsVariableInitializedbn2/beta*
_class
loc:@bn2/beta*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedbn2/moving_mean*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedbn2/moving_variance*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedbn2/moving_mean/biased*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedbn2/moving_mean/local_step*"
_class
loc:@bn2/moving_mean*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedbn2/moving_variance/biased*&
_class
loc:@bn2/moving_variance*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedbn2/moving_variance/local_step*
dtype0*
_output_shapes
: *&
_class
loc:@bn2/moving_variance

IsVariableInitialized_20IsVariableInitializedoutput/kernel* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedoutput/bias*
_class
loc:@output/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
y
IsVariableInitialized_23IsVariableInitializedSGD/lr*
dtype0*
_output_shapes
: *
_class
loc:@SGD/lr

IsVariableInitialized_24IsVariableInitializedSGD/momentum*
dtype0*
_output_shapes
: *
_class
loc:@SGD/momentum

IsVariableInitialized_25IsVariableInitialized	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/SGD/Variable*(
_class
loc:@training/SGD/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/SGD/Variable_1**
_class 
loc:@training/SGD/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/SGD/Variable_2**
_class 
loc:@training/SGD/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/SGD/Variable_3**
_class 
loc:@training/SGD/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedtraining/SGD/Variable_4**
_class 
loc:@training/SGD/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedtraining/SGD/Variable_5**
_class 
loc:@training/SGD/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedtraining/SGD/Variable_6**
_class 
loc:@training/SGD/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedtraining/SGD/Variable_7**
_class 
loc:@training/SGD/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/SGD/Variable_8*
dtype0*
_output_shapes
: **
_class 
loc:@training/SGD/Variable_8

IsVariableInitialized_35IsVariableInitializedtraining/SGD/Variable_9**
_class 
loc:@training/SGD/Variable_9*
dtype0*
_output_shapes
: 
ó
initNoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^bn1/beta/Assign^bn1/gamma/Assign^bn1/moving_mean/Assign^bn1/moving_mean/biased/Assign"^bn1/moving_mean/local_step/Assign^bn1/moving_variance/Assign"^bn1/moving_variance/biased/Assign&^bn1/moving_variance/local_step/Assign^bn2/beta/Assign^bn2/gamma/Assign^bn2/moving_mean/Assign^bn2/moving_mean/biased/Assign"^bn2/moving_mean/local_step/Assign^bn2/moving_variance/Assign"^bn2/moving_variance/biased/Assign&^bn2/moving_variance/local_step/Assign^fc1/bias/Assign^fc1/kernel/Assign^fc2/bias/Assign^fc2/kernel/Assign^output/bias/Assign^output/kernel/Assign^training/SGD/Variable/Assign^training/SGD/Variable_1/Assign^training/SGD/Variable_2/Assign^training/SGD/Variable_3/Assign^training/SGD/Variable_4/Assign^training/SGD/Variable_5/Assign^training/SGD/Variable_6/Assign^training/SGD/Variable_7/Assign^training/SGD/Variable_8/Assign^training/SGD/Variable_9/Assign
^
PlaceholderPlaceholder*
shape:	/*
dtype0*
_output_shapes
:	/

AssignAssign
fc1/kernelPlaceholder*
use_locking( *
T0*
_class
loc:@fc1/kernel*
validate_shape(*
_output_shapes
:	/
X
Placeholder_1Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_1Assignfc1/biasPlaceholder_1*
use_locking( *
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
X
Placeholder_2Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_2Assign	bn1/gammaPlaceholder_2*
T0*
_class
loc:@bn1/gamma*
validate_shape(*
_output_shapes	
:*
use_locking( 
X
Placeholder_3Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_3Assignbn1/betaPlaceholder_3*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@bn1/beta
X
Placeholder_4Placeholder*
shape:*
dtype0*
_output_shapes	
:
Ľ
Assign_4Assignbn1/moving_meanPlaceholder_4*
use_locking( *
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes	
:
X
Placeholder_5Placeholder*
dtype0*
_output_shapes	
:*
shape:
­
Assign_5Assignbn1/moving_variancePlaceholder_5*
use_locking( *
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes	
:
b
Placeholder_6Placeholder*
dtype0* 
_output_shapes
:
*
shape:

 
Assign_6Assign
fc2/kernelPlaceholder_6*
validate_shape(* 
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@fc2/kernel
X
Placeholder_7Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_7Assignfc2/biasPlaceholder_7*
use_locking( *
T0*
_class
loc:@fc2/bias*
validate_shape(*
_output_shapes	
:
X
Placeholder_8Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_8Assign	bn2/gammaPlaceholder_8*
use_locking( *
T0*
_class
loc:@bn2/gamma*
validate_shape(*
_output_shapes	
:
X
Placeholder_9Placeholder*
dtype0*
_output_shapes	
:*
shape:

Assign_9Assignbn2/betaPlaceholder_9*
use_locking( *
T0*
_class
loc:@bn2/beta*
validate_shape(*
_output_shapes	
:
Y
Placeholder_10Placeholder*
dtype0*
_output_shapes	
:*
shape:
§
	Assign_10Assignbn2/moving_meanPlaceholder_10*
use_locking( *
T0*"
_class
loc:@bn2/moving_mean*
validate_shape(*
_output_shapes	
:
Y
Placeholder_11Placeholder*
dtype0*
_output_shapes	
:*
shape:
Ż
	Assign_11Assignbn2/moving_variancePlaceholder_11*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking( 
a
Placeholder_12Placeholder*
dtype0*
_output_shapes
:	*
shape:	
§
	Assign_12Assignoutput/kernelPlaceholder_12*
validate_shape(*
_output_shapes
:	*
use_locking( *
T0* 
_class
loc:@output/kernel
W
Placeholder_13Placeholder*
shape:*
dtype0*
_output_shapes
:

	Assign_13Assignoutput/biasPlaceholder_13*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_a5dc67972e5040ba978bbb3f9e11e018/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Š
save/SaveV2/tensor_namesConst*Ü
valueŇBĎ$B	SGD/decayBSGD/iterationsBSGD/lrBSGD/momentumBbn1/betaB	bn1/gammaBbn1/moving_meanBbn1/moving_mean/biasedBbn1/moving_mean/local_stepBbn1/moving_varianceBbn1/moving_variance/biasedBbn1/moving_variance/local_stepBbn2/betaB	bn2/gammaBbn2/moving_meanBbn2/moving_mean/biasedBbn2/moving_mean/local_stepBbn2/moving_varianceBbn2/moving_variance/biasedBbn2/moving_variance/local_stepBfc1/biasB
fc1/kernelBfc2/biasB
fc2/kernelBoutput/biasBoutput/kernelBtraining/SGD/VariableBtraining/SGD/Variable_1Btraining/SGD/Variable_2Btraining/SGD/Variable_3Btraining/SGD/Variable_4Btraining/SGD/Variable_5Btraining/SGD/Variable_6Btraining/SGD/Variable_7Btraining/SGD/Variable_8Btraining/SGD/Variable_9*
dtype0*
_output_shapes
:$
Ť
save/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
Ţ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices	SGD/decaySGD/iterationsSGD/lrSGD/momentumbn1/beta	bn1/gammabn1/moving_meanbn1/moving_mean/biasedbn1/moving_mean/local_stepbn1/moving_variancebn1/moving_variance/biasedbn1/moving_variance/local_stepbn2/beta	bn2/gammabn2/moving_meanbn2/moving_mean/biasedbn2/moving_mean/local_stepbn2/moving_variancebn2/moving_variance/biasedbn2/moving_variance/local_stepfc1/bias
fc1/kernelfc2/bias
fc2/kerneloutput/biasoutput/kerneltraining/SGD/Variabletraining/SGD/Variable_1training/SGD/Variable_2training/SGD/Variable_3training/SGD/Variable_4training/SGD/Variable_5training/SGD/Variable_6training/SGD/Variable_7training/SGD/Variable_8training/SGD/Variable_9*2
dtypes(
&2$	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
Ź
save/RestoreV2/tensor_namesConst*Ü
valueŇBĎ$B	SGD/decayBSGD/iterationsBSGD/lrBSGD/momentumBbn1/betaB	bn1/gammaBbn1/moving_meanBbn1/moving_mean/biasedBbn1/moving_mean/local_stepBbn1/moving_varianceBbn1/moving_variance/biasedBbn1/moving_variance/local_stepBbn2/betaB	bn2/gammaBbn2/moving_meanBbn2/moving_mean/biasedBbn2/moving_mean/local_stepBbn2/moving_varianceBbn2/moving_variance/biasedBbn2/moving_variance/local_stepBfc1/biasB
fc1/kernelBfc2/biasB
fc2/kernelBoutput/biasBoutput/kernelBtraining/SGD/VariableBtraining/SGD/Variable_1Btraining/SGD/Variable_2Btraining/SGD/Variable_3Btraining/SGD/Variable_4Btraining/SGD/Variable_5Btraining/SGD/Variable_6Btraining/SGD/Variable_7Btraining/SGD/Variable_8Btraining/SGD/Variable_9*
dtype0*
_output_shapes
:$
Ž
save/RestoreV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
Â
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	

save/AssignAssign	SGD/decaysave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/decay
Ś
save/Assign_1AssignSGD/iterationssave/RestoreV2:1*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: *
use_locking(

save/Assign_2AssignSGD/lrsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
˘
save/Assign_3AssignSGD/momentumsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 

save/Assign_4Assignbn1/betasave/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@bn1/beta
Ą
save/Assign_5Assign	bn1/gammasave/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@bn1/gamma
­
save/Assign_6Assignbn1/moving_meansave/RestoreV2:6*
use_locking(*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes	
:
´
save/Assign_7Assignbn1/moving_mean/biasedsave/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes	
:
ł
save/Assign_8Assignbn1/moving_mean/local_stepsave/RestoreV2:8*
T0*"
_class
loc:@bn1/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(
ľ
save/Assign_9Assignbn1/moving_variancesave/RestoreV2:9*
use_locking(*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes	
:
ž
save/Assign_10Assignbn1/moving_variance/biasedsave/RestoreV2:10*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
˝
save/Assign_11Assignbn1/moving_variance/local_stepsave/RestoreV2:11*
use_locking(*
T0*&
_class
loc:@bn1/moving_variance*
validate_shape(*
_output_shapes
: 
Ą
save/Assign_12Assignbn2/betasave/RestoreV2:12*
use_locking(*
T0*
_class
loc:@bn2/beta*
validate_shape(*
_output_shapes	
:
Ł
save/Assign_13Assign	bn2/gammasave/RestoreV2:13*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@bn2/gamma
Ż
save/Assign_14Assignbn2/moving_meansave/RestoreV2:14*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@bn2/moving_mean
ś
save/Assign_15Assignbn2/moving_mean/biasedsave/RestoreV2:15*
T0*"
_class
loc:@bn2/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save/Assign_16Assignbn2/moving_mean/local_stepsave/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@bn2/moving_mean*
validate_shape(*
_output_shapes
: 
ˇ
save/Assign_17Assignbn2/moving_variancesave/RestoreV2:17*
use_locking(*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes	
:
ž
save/Assign_18Assignbn2/moving_variance/biasedsave/RestoreV2:18*
use_locking(*
T0*&
_class
loc:@bn2/moving_variance*
validate_shape(*
_output_shapes	
:
˝
save/Assign_19Assignbn2/moving_variance/local_stepsave/RestoreV2:19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@bn2/moving_variance
Ą
save/Assign_20Assignfc1/biassave/RestoreV2:20*
use_locking(*
T0*
_class
loc:@fc1/bias*
validate_shape(*
_output_shapes	
:
Š
save/Assign_21Assign
fc1/kernelsave/RestoreV2:21*
T0*
_class
loc:@fc1/kernel*
validate_shape(*
_output_shapes
:	/*
use_locking(
Ą
save/Assign_22Assignfc2/biassave/RestoreV2:22*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@fc2/bias
Ş
save/Assign_23Assign
fc2/kernelsave/RestoreV2:23*
use_locking(*
T0*
_class
loc:@fc2/kernel*
validate_shape(* 
_output_shapes
:

Ś
save/Assign_24Assignoutput/biassave/RestoreV2:24*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:
Ż
save/Assign_25Assignoutput/kernelsave/RestoreV2:25*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes
:	
ż
save/Assign_26Assigntraining/SGD/Variablesave/RestoreV2:26*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes
:	/*
use_locking(
ż
save/Assign_27Assigntraining/SGD/Variable_1save/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1
ż
save/Assign_28Assigntraining/SGD/Variable_2save/RestoreV2:28*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*
_output_shapes	
:*
use_locking(
ż
save/Assign_29Assigntraining/SGD/Variable_3save/RestoreV2:29*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes	
:
Ä
save/Assign_30Assigntraining/SGD/Variable_4save/RestoreV2:30*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(* 
_output_shapes
:

ż
save/Assign_31Assigntraining/SGD/Variable_5save/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_5
ż
save/Assign_32Assigntraining/SGD/Variable_6save/RestoreV2:32*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_6*
validate_shape(*
_output_shapes	
:
ż
save/Assign_33Assigntraining/SGD/Variable_7save/RestoreV2:33*
T0**
_class 
loc:@training/SGD/Variable_7*
validate_shape(*
_output_shapes	
:*
use_locking(
Ă
save/Assign_34Assigntraining/SGD/Variable_8save/RestoreV2:34*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_8
ž
save/Assign_35Assigntraining/SGD/Variable_9save/RestoreV2:35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_9
ň
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"Â
trainable_variablesŞ§
L
fc1/kernel:0fc1/kernel/Assignfc1/kernel/read:02fc1/random_uniform:08
=

fc1/bias:0fc1/bias/Assignfc1/bias/read:02fc1/Const:08
@
bn1/gamma:0bn1/gamma/Assignbn1/gamma/read:02bn1/Const:08
?

bn1/beta:0bn1/beta/Assignbn1/beta/read:02bn1/Const_1:08
T
bn1/moving_mean:0bn1/moving_mean/Assignbn1/moving_mean/read:02bn1/Const_2:08
`
bn1/moving_variance:0bn1/moving_variance/Assignbn1/moving_variance/read:02bn1/Const_3:08
L
fc2/kernel:0fc2/kernel/Assignfc2/kernel/read:02fc2/random_uniform:08
=

fc2/bias:0fc2/bias/Assignfc2/bias/read:02fc2/Const:08
@
bn2/gamma:0bn2/gamma/Assignbn2/gamma/read:02bn2/Const:08
?

bn2/beta:0bn2/beta/Assignbn2/beta/read:02bn2/Const_1:08
T
bn2/moving_mean:0bn2/moving_mean/Assignbn2/moving_mean/read:02bn2/Const_2:08
`
bn2/moving_variance:0bn2/moving_variance/Assignbn2/moving_variance/read:02bn2/Const_3:08
X
output/kernel:0output/kernel/Assignoutput/kernel/read:02output/random_uniform:08
I
output/bias:0output/bias/Assignoutput/bias/read:02output/Const:08
b
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:08
B
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:08
Z
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:08
N
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:08
m
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:08
u
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:08
u
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:08
u
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:08
u
training/SGD/Variable_4:0training/SGD/Variable_4/Assigntraining/SGD/Variable_4/read:02training/SGD/zeros_4:08
u
training/SGD/Variable_5:0training/SGD/Variable_5/Assigntraining/SGD/Variable_5/read:02training/SGD/zeros_5:08
u
training/SGD/Variable_6:0training/SGD/Variable_6/Assigntraining/SGD/Variable_6/read:02training/SGD/zeros_6:08
u
training/SGD/Variable_7:0training/SGD/Variable_7/Assigntraining/SGD/Variable_7/read:02training/SGD/zeros_7:08
u
training/SGD/Variable_8:0training/SGD/Variable_8/Assigntraining/SGD/Variable_8/read:02training/SGD/zeros_8:08
u
training/SGD/Variable_9:0training/SGD/Variable_9/Assigntraining/SGD/Variable_9/read:02training/SGD/zeros_9:08"Ö
cond_contextĹÂ

bn1/cond/cond_textbn1/cond/pred_id:0bn1/cond/switch_t:0 *Â
bn1/batchnorm/add_1:0
bn1/cond/Switch_1:0
bn1/cond/Switch_1:1
bn1/cond/pred_id:0
bn1/cond/switch_t:0(
bn1/cond/pred_id:0bn1/cond/pred_id:0,
bn1/batchnorm/add_1:0bn1/cond/Switch_1:1

bn1/cond/cond_text_1bn1/cond/pred_id:0bn1/cond/switch_f:0*Ő
bn1/beta/read:0
bn1/cond/batchnorm/Rsqrt:0
bn1/cond/batchnorm/add/Switch:0
bn1/cond/batchnorm/add/y:0
bn1/cond/batchnorm/add:0
bn1/cond/batchnorm/add_1:0
bn1/cond/batchnorm/mul/Switch:0
bn1/cond/batchnorm/mul:0
!bn1/cond/batchnorm/mul_1/Switch:0
bn1/cond/batchnorm/mul_1:0
!bn1/cond/batchnorm/mul_2/Switch:0
bn1/cond/batchnorm/mul_2:0
bn1/cond/batchnorm/sub/Switch:0
bn1/cond/batchnorm/sub:0
bn1/cond/pred_id:0
bn1/cond/switch_f:0
bn1/gamma/read:0
bn1/moving_mean/read:0
bn1/moving_variance/read:0
fc1/BiasAdd:0(
bn1/cond/pred_id:0bn1/cond/pred_id:03
bn1/gamma/read:0bn1/cond/batchnorm/mul/Switch:0=
bn1/moving_variance/read:0bn1/cond/batchnorm/add/Switch:0;
bn1/moving_mean/read:0!bn1/cond/batchnorm/mul_2/Switch:02
fc1/BiasAdd:0!bn1/cond/batchnorm/mul_1/Switch:02
bn1/beta/read:0bn1/cond/batchnorm/sub/Switch:0

bn2/cond/cond_textbn2/cond/pred_id:0bn2/cond/switch_t:0 *Â
bn2/batchnorm/add_1:0
bn2/cond/Switch_1:0
bn2/cond/Switch_1:1
bn2/cond/pred_id:0
bn2/cond/switch_t:0,
bn2/batchnorm/add_1:0bn2/cond/Switch_1:1(
bn2/cond/pred_id:0bn2/cond/pred_id:0

bn2/cond/cond_text_1bn2/cond/pred_id:0bn2/cond/switch_f:0*Ő
bn2/beta/read:0
bn2/cond/batchnorm/Rsqrt:0
bn2/cond/batchnorm/add/Switch:0
bn2/cond/batchnorm/add/y:0
bn2/cond/batchnorm/add:0
bn2/cond/batchnorm/add_1:0
bn2/cond/batchnorm/mul/Switch:0
bn2/cond/batchnorm/mul:0
!bn2/cond/batchnorm/mul_1/Switch:0
bn2/cond/batchnorm/mul_1:0
!bn2/cond/batchnorm/mul_2/Switch:0
bn2/cond/batchnorm/mul_2:0
bn2/cond/batchnorm/sub/Switch:0
bn2/cond/batchnorm/sub:0
bn2/cond/pred_id:0
bn2/cond/switch_f:0
bn2/gamma/read:0
bn2/moving_mean/read:0
bn2/moving_variance/read:0
fc2/BiasAdd:03
bn2/gamma/read:0bn2/cond/batchnorm/mul/Switch:0(
bn2/cond/pred_id:0bn2/cond/pred_id:02
bn2/beta/read:0bn2/cond/batchnorm/sub/Switch:02
fc2/BiasAdd:0!bn2/cond/batchnorm/mul_1/Switch:0;
bn2/moving_mean/read:0!bn2/cond/batchnorm/mul_2/Switch:0=
bn2/moving_variance/read:0bn2/cond/batchnorm/add/Switch:0"ř
	variablesęç
L
fc1/kernel:0fc1/kernel/Assignfc1/kernel/read:02fc1/random_uniform:08
=

fc1/bias:0fc1/bias/Assignfc1/bias/read:02fc1/Const:08
@
bn1/gamma:0bn1/gamma/Assignbn1/gamma/read:02bn1/Const:08
?

bn1/beta:0bn1/beta/Assignbn1/beta/read:02bn1/Const_1:08
T
bn1/moving_mean:0bn1/moving_mean/Assignbn1/moving_mean/read:02bn1/Const_2:08
`
bn1/moving_variance:0bn1/moving_variance/Assignbn1/moving_variance/read:02bn1/Const_3:08

bn1/moving_mean/biased:0bn1/moving_mean/biased/Assignbn1/moving_mean/biased/read:02+bn1/AssignMovingAvg/bn1/moving_mean/zeros:0

bn1/moving_mean/local_step:0!bn1/moving_mean/local_step/Assign!bn1/moving_mean/local_step/read:02.bn1/moving_mean/local_step/Initializer/zeros:0

bn1/moving_variance/biased:0!bn1/moving_variance/biased/Assign!bn1/moving_variance/biased/read:021bn1/AssignMovingAvg_1/bn1/moving_variance/zeros:0
¤
 bn1/moving_variance/local_step:0%bn1/moving_variance/local_step/Assign%bn1/moving_variance/local_step/read:022bn1/moving_variance/local_step/Initializer/zeros:0
L
fc2/kernel:0fc2/kernel/Assignfc2/kernel/read:02fc2/random_uniform:08
=

fc2/bias:0fc2/bias/Assignfc2/bias/read:02fc2/Const:08
@
bn2/gamma:0bn2/gamma/Assignbn2/gamma/read:02bn2/Const:08
?

bn2/beta:0bn2/beta/Assignbn2/beta/read:02bn2/Const_1:08
T
bn2/moving_mean:0bn2/moving_mean/Assignbn2/moving_mean/read:02bn2/Const_2:08
`
bn2/moving_variance:0bn2/moving_variance/Assignbn2/moving_variance/read:02bn2/Const_3:08

bn2/moving_mean/biased:0bn2/moving_mean/biased/Assignbn2/moving_mean/biased/read:02+bn2/AssignMovingAvg/bn2/moving_mean/zeros:0

bn2/moving_mean/local_step:0!bn2/moving_mean/local_step/Assign!bn2/moving_mean/local_step/read:02.bn2/moving_mean/local_step/Initializer/zeros:0

bn2/moving_variance/biased:0!bn2/moving_variance/biased/Assign!bn2/moving_variance/biased/read:021bn2/AssignMovingAvg_1/bn2/moving_variance/zeros:0
¤
 bn2/moving_variance/local_step:0%bn2/moving_variance/local_step/Assign%bn2/moving_variance/local_step/read:022bn2/moving_variance/local_step/Initializer/zeros:0
X
output/kernel:0output/kernel/Assignoutput/kernel/read:02output/random_uniform:08
I
output/bias:0output/bias/Assignoutput/bias/read:02output/Const:08
b
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:08
B
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:08
Z
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:08
N
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:08
m
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:08
u
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:08
u
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:08
u
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:08
u
training/SGD/Variable_4:0training/SGD/Variable_4/Assigntraining/SGD/Variable_4/read:02training/SGD/zeros_4:08
u
training/SGD/Variable_5:0training/SGD/Variable_5/Assigntraining/SGD/Variable_5/read:02training/SGD/zeros_5:08
u
training/SGD/Variable_6:0training/SGD/Variable_6/Assigntraining/SGD/Variable_6/read:02training/SGD/zeros_6:08
u
training/SGD/Variable_7:0training/SGD/Variable_7/Assigntraining/SGD/Variable_7/read:02training/SGD/zeros_7:08
u
training/SGD/Variable_8:0training/SGD/Variable_8/Assigntraining/SGD/Variable_8/read:02training/SGD/zeros_8:08
u
training/SGD/Variable_9:0training/SGD/Variable_9/Assigntraining/SGD/Variable_9/read:02training/SGD/zeros_9:08*¤
serving_default
5
input_tensor%
input_tensor:0˙˙˙˙˙˙˙˙˙/;
output/BiasAdd:0'
output/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict