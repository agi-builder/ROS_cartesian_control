#!/usr/bin/env python
# coding: UTF-8
import sys
l1111lll_opy_ = sys.version_info [0] == 2
l1l1ll1l_opy_ = 2048
l11l11l1_opy_ = 7
def l11ll_opy_ (l1111l_opy_):
    global l1ll111l_opy_
    l1l11l1l_opy_ = ord (l1111l_opy_ [-1])
    l111l11l_opy_ = l1111l_opy_ [:-1]
    l1lllll1_opy_ = l1l11l1l_opy_ % len (l111l11l_opy_)
    l1lll11_opy_ = l111l11l_opy_ [:l1lllll1_opy_] + l111l11l_opy_ [l1lllll1_opy_:]
    if l1111lll_opy_:
        l1ll11_opy_ = unicode () .join ([unichr (ord (char) - l1l1ll1l_opy_ - (l1l11_opy_ + l1l11l1l_opy_) % l11l11l1_opy_) for l1l11_opy_, char in enumerate (l1lll11_opy_)])
    else:
        l1ll11_opy_ = str () .join ([chr (ord (char) - l1l1ll1l_opy_ - (l1l11_opy_ + l1l11l1l_opy_) % l11l11l1_opy_) for l1l11_opy_, char in enumerate (l1lll11_opy_)])
    return eval (l1ll11_opy_)
import rospy
import numpy
import tf
import time
from sensor_msgs.msg import JointState
from cartesian_control.msg import CartesianCommand
import geometry_msgs.msg
from geometry_msgs.msg import Transform
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker
from interactive_markers.interactive_marker_server import *
from urdf_parser_py.urdf import URDF
def l1l1l111l_opy_(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]
    return t
def is_same_trans(l1l1ll111_opy_, l1ll11ll1_opy_):
    return (abs(numpy.subtract(l1l1ll111_opy_, l1ll11ll1_opy_)) < 10e-3).all()
def is_same(matrix0, matrix1):
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1, 0, 1e-2)
def l1l11lll1_opy_(T):
    t = Transform()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.translation.x = position[0]
    t.translation.y = position[1]
    t.translation.z = position[2]
    t.rotation.x = orientation[0]
    t.rotation.y = orientation[1]
    t.rotation.z = orientation[2]
    t.rotation.w = orientation[3]
    return t
class CartesianGrader(object):
    #l1l1l1lll_opy_
    def __init__(self):
        self.l1l1ll1ll_opy_ = rospy.Publisher(l11ll_opy_ (u"ࠦ࠴ࡰ࡯ࡪࡰࡷࡣࡨࡵ࡭࡮ࡣࡱࡨࠧ࣋"), JointState, queue_size=1)
        #Publisher to l1l11l1ll_opy_ l1l1l1l11_opy_
        self.l1l1ll1l1_opy_ = rospy.Publisher(l11ll_opy_ (u"ࠧ࠵ࡣࡢࡴࡷࡩࡸ࡯ࡡ࡯ࡡࡦࡳࡲࡳࡡ࡯ࡦࠥ࣌"), CartesianCommand, queue_size=1)
        self.l1l1lll11_opy_ = rospy.Publisher(l11ll_opy_ (u"ࠨ࠯ࡪ࡭ࡢࡧࡴࡳ࡭ࡢࡰࡧࠦ࣍"), Transform, queue_size=1)
	rospy.Subscriber(l11ll_opy_ (u"ࠧ࠰࡬ࡲ࡭ࡳࡺ࡟ࡴࡶࡤࡸࡪࡹࠧ࣎"), JointState, self.l1l1l1111_opy_)
        self.listener = tf.TransformListener()
        self.robot = URDF.from_parameter_server()
        self.l11l11l11_opy_ = 0
        self.joint_names = []
        self.l11ll1ll1_opy_ = []
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (l111l111l_opy_, l111l1111_opy_) = self.robot.child_map[link][0]
            l111l11l1_opy_ = self.robot.joint_map[l111l111l_opy_]
            if l111l11l1_opy_.type != l11ll_opy_ (u"ࠨࡨ࡬ࡼࡪࡪ࣏ࠧ"):
                self.l11l11l11_opy_ = self.l11l11l11_opy_ + 1
                self.joint_names.append(l111l11l1_opy_.name)
                self.l11ll1ll1_opy_.append(l111l11l1_opy_.axis)
            link = l111l1111_opy_
        self.l111l11ll_opy_ = link
    def l1l1l1111_opy_(self, msg):
        self.l1l1l1ll1_opy_ = msg.position[0]
    def reset_robot(self):
        cmd = JointState()
	if self.robot.name == l11ll_opy_ (u"ࠩࡸࡶ࠺࣐࠭"):
            cmd.position.append(1.9)
            cmd.position.append(0.65)
            cmd.position.append(1.5)
            cmd.position.append(1.15)
            cmd.position.append(0)
            cmd.position.append(0.3)
        else:
            cmd.position.append(0.0)
            cmd.position.append(1.0)
            cmd.position.append(0.0)
            cmd.position.append(1.0)
            cmd.position.append(0.0)
            cmd.position.append(0.0)
            cmd.position.append(0.0)
        self.l1l1ll1ll_opy_.publish(cmd)
        rospy.sleep(1.0)
    def go_to_ik_pose(self, name, T, timeout):
        msg = l1l11lll1_opy_(T)
        start_time = time.time()
        l1l11ll1l_opy_ = False
	self.l1l1lll11_opy_.publish(msg)
        while not l1l11ll1l_opy_ and not rospy.is_shutdown():
            try:
                (l1l1lllll_opy_,rot) = self.listener.lookupTransform(self.robot.get_root(),self.l111l11ll_opy_,
                                                            rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print l11ll_opy_ (u"ࠥࡘࡋࠦࡅࡹࡥࡨࡴࡹ࡯࡯࡯࣑ࠣࠥ")
                continue
            l1l1l1l1l_opy_ = numpy.dot(tf.transformations.translation_matrix(l1l1lllll_opy_),
                           tf.transformations.quaternion_matrix(rot))
            if (is_same_trans(tf.transformations.translation_from_matrix(T), tf.transformations.translation_from_matrix(l1l1l1l1l_opy_))):
               #print name + l11ll_opy_ (u"ࠦ࠿ࠦࡐࡂࡕࡖࡉࡉࠨ࣒")
               l1l11ll1l_opy_ = True
               return 1
            if (time.time() - start_time > timeout) :
                #print name + l11ll_opy_ (u"ࠧࡀࠠࡓࡱࡥࡳࡹࠦࡴࡰࡱ࡮ࠤࡹࡵ࡯ࠡ࡮ࡲࡲ࡬ࠦࡴࡰࠢࡵࡩࡦࡩࡨࠡࡦࡨࡷ࡮ࡸࡥࡥࠢࡳࡳࡸ࡫࣓ࠢ")
                #print l11ll_opy_ (u"ࠨࡒࡰࡤࡲࡸࠥࡺ࡯ࡰ࡭ࠣࡸࡴࡵࠠ࡭ࡱࡱ࡫ࠥࡺ࡯ࠡࡴࡨࡥࡨ࡮ࠠࡥࡧࡶ࡭ࡷ࡫ࡤࠡࡲࡲࡷࡪ࠴ࠠࡈࡴࡤࡨࡪࡸࠠࡵ࡫ࡰࡩࡩࠦ࡯ࡶࡶࠥࣔ")
                l1l11ll1l_opy_ = True
            else:
                rospy.sleep(0.1)
        return 0
    def go_to_pose(self, name, T, secondary_objective, q0_target, timeout, points):
        msg = l1l11lll1_opy_(T)
	l1l1l11ll_opy_ = CartesianCommand()
	l1l1l11ll_opy_.x_target = msg
	l1l1l11ll_opy_.secondary_objective = secondary_objective
	l1l1l11ll_opy_.q0_target = q0_target
        start_time = time.time()
        l1l11ll1l_opy_ = False
        index = 0
        while not l1l11ll1l_opy_ and not rospy.is_shutdown():
            index += 1
            self.l1l1ll1l1_opy_.publish(l1l1l11ll_opy_)
            try:
                (l1l1lllll_opy_,rot) = self.listener.lookupTransform(self.robot.get_root(),self.l111l11ll_opy_,
                                                            rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print l11ll_opy_ (u"ࠢࡕࡈࠣࡉࡽࡩࡥࡱࡶ࡬ࡳࡳࠧࠢࣕ")
                continue
            l1l1l1l1l_opy_ = numpy.dot(tf.transformations.translation_matrix(l1l1lllll_opy_),
                           tf.transformations.quaternion_matrix(rot))
            if (is_same(T, l1l1l1l1l_opy_)):
		if secondary_objective:
                  if self.robot.name == l11ll_opy_ (u"ࠨࡷࡵ࠹ࠬࣖ") and index>30:
                        return points
                    	l1l11ll1l_opy_ = True
                  else:
		    if abs(self.l1l1l1ll1_opy_-q0_target)< 10e-3:
			#print name + l11ll_opy_ (u"ࠤ࠽ࠤࡕࡇࡓࡔࡇࡇࠦࣗ")
                        return points
                    	l1l11ll1l_opy_ = True
		else:
                    #print name + l11ll_opy_ (u"ࠥ࠾ࠥࡖࡁࡔࡕࡈࡈࠧࣘ")
                    return points
                    l1l11ll1l_opy_ = True
            if (time.time() - start_time > timeout) :
                #print name + l11ll_opy_ (u"ࠦ࠿ࠦࡒࡰࡤࡲࡸࠥࡺ࡯ࡰ࡭ࠣࡸࡴࡵࠠ࡭ࡱࡱ࡫ࠥࡺ࡯ࠡࡴࡨࡥࡨ࡮ࠠࡥࡧࡶ࡭ࡷ࡫ࡤࠡࡲࡲࡷࡪࠨࣙ")
                #print l11ll_opy_ (u"ࠧࡘ࡯ࡣࡱࡷࠤࡹࡵ࡯࡬ࠢࡷࡳࡴࠦ࡬ࡰࡰࡪࠤࡹࡵࠠࡳࡧࡤࡧ࡭ࠦࡤࡦࡵ࡬ࡶࡪࡪࠠࡱࡱࡶࡩ࠳ࠦࡇࡳࡣࡧࡩࡷࠦࡴࡪ࡯ࡨࡨࠥࡵࡵࡵࠤࣚ")
                return 0
                l1l11ll1l_opy_ = True
            else:
                rospy.sleep(0.1)
        return 0