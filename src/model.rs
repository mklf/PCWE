extern crate rand;
#[cfg(feature="blas")]
use blas_sys::c;
use libc;
use std::sync::Arc;
use std::mem::size_of;
use matrix::Matrix;
use {MAX_SIGMOID, SIGMOID_TABLE_SIZE, LOG_TABLE_SIZE,ACCEPT_TABLE_SIZE};
const SIGMOID_TABLE_SIZE_F: f32 = SIGMOID_TABLE_SIZE as f32;
const LOG_TABLE_SIZE_F: f32 = LOG_TABLE_SIZE as f32;
use std::ptr;

fn init_accept_table() ->[f32;ACCEPT_TABLE_SIZE]{
    let mut accept_table = [0f32;ACCEPT_TABLE_SIZE];
    for i in 0..ACCEPT_TABLE_SIZE {
        accept_table[i] = 1.-(i as f32/ ACCEPT_TABLE_SIZE as f32)
    }
    accept_table
}


fn init_sigmoid_table() -> [f32; SIGMOID_TABLE_SIZE + 1] {
    let mut sigmoid_table = [0f32; SIGMOID_TABLE_SIZE + 1];
    for i in 0..SIGMOID_TABLE_SIZE + 1 {
        let x = (i as f32 * 2. * MAX_SIGMOID) / SIGMOID_TABLE_SIZE_F - MAX_SIGMOID;
        sigmoid_table[i] = 1.0 / (1.0 + (-x).exp());
    }
    sigmoid_table
}
fn init_log_table() -> [f32; LOG_TABLE_SIZE + 1] {
    let mut log_table = [0f32; LOG_TABLE_SIZE + 1];
    for i in 0..LOG_TABLE_SIZE + 1 {
        let x = (i as f32 + 1e-5) / LOG_TABLE_SIZE_F;
        log_table[i] = x.ln();
    }
    log_table
}

pub struct Model<'a> {
    pub input: &'a mut Matrix,
    output: &'a mut Matrix,
    lr: f32,
    neg: usize,
    grad_: Vec<f32>,
    hidden:Vec<f32>,
    neg_pos: usize,
    sigmoid_table: [f32; SIGMOID_TABLE_SIZE + 1],
    log_table: [f32; LOG_TABLE_SIZE + 1],
    negative_table: Arc<Vec<usize>>,
    loss: f64,
    nsamples: u64,
    grad_ptr:*mut f32,
    hidden_ptr:*mut f32,
    accept_table:[f32;ACCEPT_TABLE_SIZE]
}
impl<'a> Model<'a> {
    pub fn new(input: &'a mut Matrix,
               output: &'a mut Matrix,
               dim: usize,
               lr: f32,
               // tid: u32,
               neg: usize,
               neg_table: Arc<Vec<usize>>)
               -> Model<'a> {
        let mut m = Model {
            input: input,
            output: output,
            lr: lr,
            neg: neg,
            grad_: vec![0f32;dim],
            neg_pos: 0,
            sigmoid_table: init_sigmoid_table(),
            log_table: init_log_table(),
            negative_table: neg_table,
            loss: 0.,
            nsamples: 0,
            hidden:vec![0f32;dim],
            hidden_ptr:ptr::null_mut(),
            grad_ptr:ptr::null_mut(),
            accept_table:init_accept_table(),
        };
        m.hidden_ptr = unsafe {m.hidden.get_unchecked_mut(0)};
        m.grad_ptr = unsafe{m.grad_.get_unchecked_mut(0)};
        m
    }
    #[inline]
    pub fn accept_probability(&self,count:usize)->f32{
        self.accept_table[count]
    }
    #[inline]
    fn log(&self, x: f32) -> f32 {
        if x > 1.0 {
            x
        } else {
            let i = (x * (LOG_TABLE_SIZE_F)) as usize;
            unsafe { *self.log_table.get_unchecked(i) }
        }
    }
    #[inline]
    fn sigmoid(&self, x: f32) -> f32 {
        if x < -MAX_SIGMOID {
            0f32
        } else if x > MAX_SIGMOID {
            1f32
        } else {
            let i = (x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE_F / MAX_SIGMOID / 2.;
            unsafe { *self.sigmoid_table.get_unchecked(i as usize) }
        }
        //return 1./(1.+(-x).exp())
    }
    #[inline]
    pub fn get_loss(&self) -> f64 {
        self.loss / self.nsamples as f64
    }
    #[inline(always)]
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    #[inline(always)]
    pub fn get_lr(&self) -> f32 {
        self.lr
    }
    #[inline(always)]
    fn compute_hidden(&mut self,input:&Vec<usize>){
        Model::zero_vector(&mut self.hidden);
        for index in input{
            let sub = self.input.get_row(*index);
            Model::add_mul_row(&mut self.hidden,sub,1.);
        }
        let inv_size = 1./(input.len() as f32);
        /*for i in 0..self.dim as isize{
            unsafe{ *self.hidden_ptr.offset(i) *= inv_size;}
        }*/
        for v in self.hidden.iter_mut(){
            *v *=inv_size;
        }
    }

    fn binary_losgistic(&mut self, target: usize, label: i32) -> f32 {
        let sum = self.output.dot_row(self.hidden_ptr, target);
        let score = self.sigmoid(sum);
        let alpha = self.lr * (label as f32 - score);
        let tar_emb = self.output.get_row(target);
        Model::add_mul_row(&mut self.grad_,tar_emb, alpha);
        self.output.add_row(self.hidden_ptr, target, alpha);
        if label == 1 {
            let loss = -self.log(score);
            loss
        } else {
            let loss= -self.log(1.0 - score);
            loss
        }
    }
    #[inline(always)]
    pub fn update(&mut self, input: &Vec<usize>, target: usize) {
        if input.len()==1 {
            self.hidden_ptr = self.input.get_row(input[0]);
        }else {
            self.compute_hidden(input);
        }
        self.loss += self.negative_sampling(target);
        self.nsamples += 1;
        /*println!("{:?}",self.grad_);
        use std::process;
        process::exit(0);
        */
        for index in input{
            self.input.add_row(self.grad_ptr,*index,1.);
        }
    }

    fn negative_sampling(&mut self, target: usize) -> f64 {

        let mut loss = 0f32;
        Model::zero_vector(&mut self.grad_);

        for i in 0..self.neg + 1 {
            if i == 0 {
                loss += self.binary_losgistic( target, 1);
            } else {
                let neg_sample = self.get_negative(target);
                loss += self.binary_losgistic( neg_sample, 0);
            }
        }
        loss as f64
    }
    fn get_negative(&mut self, target: usize) -> usize {
        loop {
            let negative = self.negative_table[self.neg_pos];
            self.neg_pos = (self.neg_pos + 1) % self.negative_table.len();
            if target != negative {
                return negative;
            }
        }
    }
    #[inline(always)]
    pub fn zero_vector(v:&mut Vec<f32>){
        unsafe {
            libc::memset(v.as_mut_ptr() as *mut libc::c_void,
                         0,
                         v.len() * size_of::<f32>())
        };
    }

    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn add_mul_row(v:&mut Vec<f32>, other: *const f32, a: f32) {
        unsafe { c::cblas_saxpy(v.len() as i32, a, other, 1, v.as_mut_ptr(), 1) };
    }
    #[cfg(not(feature="blas"))]
    #[inline(always)]
    pub fn add_mul_row(v:&mut Vec<f32>, other: *const f32, a: f32) {
        for i in 0..v.len() {
            unsafe {
                *v.get_unchecked_mut(i) += a * (*other.offset(i as isize));
            }
        }
    }
}