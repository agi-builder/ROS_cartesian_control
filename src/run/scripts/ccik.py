#!/usr/bin/env python

import math
import numpy
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import tf
from threading import Thread, Lock

'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(object):
    def __init__(self):
	#Load robot from parameter server
        self.robot = URDF.from_parameter_server()

	#Subscribe to current joint state of the robot
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)

	#This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

	#This is a mutex
        self.mutex = Lock()

	#Subscribers and publishers for for cartesian control
        rospy.Subscriber('/cartesian_command', CartesianCommand, self.get_cartesian_command)
        self.velocity_pub = rospy.Publisher('/joint_velocities', JointState, queue_size=10)
        self.joint_velocity_msg = JointState()

        #Subscribers and publishers for numerical IK
        rospy.Subscriber('/ik_command', Transform, self.get_ik_command)
        self.joint_command_pub = rospy.Publisher('/joint_command', JointState, queue_size=10)
        self.joint_command_msg = JointState()

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link

    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR CARTESIAN CONTROL HERE
        joint_transforms, b_T_eecur = self.forward_kinematics(self.q_current)
        cmd_t = tf.transformations.translation_matrix((command.x_target.translation.x, command.x_target.translation.y, command.x_target.translation.z))
        cmd_r = tf.transformations.quaternion_matrix((command.x_target.rotation.x, command.x_target.rotation.y, command.x_target.rotation.z, command.x_target.rotation.w))
        b_T_eecmd = numpy.dot(cmd_t, cmd_r)
        eecur_T_eecmd = numpy.dot(tf.transformations.inverse_matrix(b_T_eecur), b_T_eecmd)
        dx = numpy.array([eecur_T_eecmd[0][3], eecur_T_eecmd[1][3], eecur_T_eecmd[2][3]])
        dx = 2*dx

        if (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5 > 0.1:
            dx = dx/(dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5/10

        angle, axis = self.rotation_from_matrix(eecur_T_eecmd)
        dori = axis*angle
        dori = dori*2

        if (dori[0]**2 + dori[1]**2 + dori[2]**2)**0.5 > 1:
            dori = dori/(dori[0]**2 + dori[1]**2 + dori[2]**2)**0.5

        v_ee = numpy.append(dx,dori)     
        J = self.get_jacobian(b_T_eecur, joint_transforms)
        Jp = numpy.linalg.pinv(J)
        q = numpy.dot(Jp,v_ee)
       

        if command.secondary_objective == True:
            q_sec = numpy.zeros((self.num_joints,1))
            q_sec[0] = (command.q0_target - self.q_current[0])
            q_null = numpy.dot(numpy.identity(self.num_joints) - numpy.dot(Jp,J), q_sec)
            for i in range(self.num_joints):
                q[i] = q[i] + 3*q_null[i]
            
        if max(abs(q)) > 1:
            q = q/max(abs(q))

        
        self.joint_velocity_msg.velocity = q
        self.joint_velocity_msg.name = self.joint_names
        self.joint_velocity_msg.position = self.q_current
        self.velocity_pub.publish(self.joint_velocity_msg)


        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_ee, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR ASSEMBLING THE CURRENT JACOBIAN HERE
        for i in range(self.num_joints):
            R = numpy.zeros((3, 3))
            t = numpy.zeros((3, 1))
            St = numpy.zeros((3, 3))
            j_T_ee = numpy.dot(tf.transformations.inverse_matrix(joint_transforms[i]), b_T_ee)
            ee_T_j = tf.transformations.inverse_matrix(j_T_ee)
            for m in range(3):
                for n in range(3):
                    R[m][n] = ee_T_j[m][n]
            for m in range(3):
                t[m] = j_T_ee[m][3]
            St[1][2] = -t[0]
            St[2][1] = t[0]
            St[0][2] = t[1]
            St[2][0] = -t[1]
            St[0][1] = -t[2]
            St[1][0] = t[2]
            Vj = numpy.hstack((R,-numpy.dot(R, St)))
            Vj = numpy.vstack((Vj, numpy.hstack((numpy.zeros((3,3)), R))))
            flag = self.joint_axes[i]    
            if flag[0] == 1:
                J[:,i] = Vj[:,3] 
            if flag[0] == -1:
                J[:,i] = -Vj[:,3]
            if flag[1] == 1:
                J[:,i] = Vj[:,4] 
            if flag[1] == -1:
                J[:,i] = -Vj[:,4]
            if flag[2] == 1:
                J[:,i] = Vj[:,5] 
            if flag[2] == -1:
                J[:,i] = -Vj[:,5]
        #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR INVERSE KINEMATICS HERE
        #joint_transforms, b_T_eecur = self.forward_kinematics(self.q_current)

        cmd_t = tf.transformations.translation_matrix((command.translation.x,command.translation.y,command.translation.z))
        cmd_r = tf.transformations.quaternion_matrix((command.rotation.x,command.rotation.y,command.rotation.z,command.rotation.w))
        b_T_eecmd = numpy.dot(cmd_t,cmd_r)

        q_c = numpy.zeros((self.num_joints))
        for i in range(self.num_joints):
            q_c[i] = random.uniform(0,2*math.pi)

        while True:
            joint_transforms_c, b_T_c = self.forward_kinematics(q_c)
            c_T_eecmd = numpy.dot(tf.transformations.inverse_matrix(b_T_c), b_T_eecmd)
            dx = numpy.array([c_T_eecmd[0][3], c_T_eecmd[1][3], c_T_eecmd[2][3]])
            angle, axis = self.rotation_from_matrix(c_T_eecmd)
            dori = axis*angle
            dx = numpy.append(dx,dori) 
            if max(abs(dx)) < 10**(-3):
                break
            J = self.get_jacobian(b_T_c, joint_transforms_c)
            Jp = numpy.linalg.pinv(J)
            dq = numpy.dot(Jp, dx)
            q_c = q_c + dq
        
        self.joint_command_msg.name = self.joint_names
        self.joint_command_msg.position = q_c
        self.joint_command_pub.publish(self.joint_command_msg)
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each joint of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.transformations.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.transformations.translation_matrix(joint.origin.xyz), tf.transformations.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]))
            T = numpy.dot(T, T_l)

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.transformations.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()


if __name__ == '__main__':
    rospy.init_node('cartesian_control_and_IK', anonymous=True)
    CCIK()
    rospy.spin()
