from src.modules.MultiTracker import MultiTracker
from src.modules.Mapper import Mapper
from src import config
import pytest

dummy_detection_cam8 = [{'bbox_xyxy': (370, 846, 450, 926), 'center':(410, 886), 'conf': 0.8, 'cls': 0}, {'bbox_xyxy': (1401, 860, 1481, 940), 'center':(1441, 900), 'conf': 0.7, 'cls': 0}]  # (1,0) and outlier
dummy_detection_cam9 = [{'bbox_xyxy': (538, 856, 618, 936), 'center':(578, 896), 'conf': 0.78, 'cls': 0}]  # (1,0)  
dummy_detection_cam6 = [{'bbox_xyxy': (2325, 125, 2385, 185), 'center':(2355, 155), 'conf': 0.78, 'cls': 0}]  # (1,0) 

dummy_batch_path_cam9 = [[], 
                            # frame 8 is outlier and should be in next path
                            [((0.8541231751441956, -1.1426388025283813), 0), ((0.8541231751441956, -1.1426388025283813), 1), ((0.8517820835113525, -1.1423827409744263), 2), ((0.849441409111023, -1.1421266794204712), 3), ((0.849441409111023, -1.1421266794204712), 4), ((0.8499664664268494, -1.130196452140808), 5), ((0.8479819893836975, -1.1220126152038574), 6), ((0.8434942960739136, -1.1175485849380493), 7), ((2.180703639984131, 0.9492424130439758), 8), ((0.8381338715553284, -1.0814005136489868), 9)], 
                            # frame 8 is outlier and should be in previous path
                            [((2.180703639984131, 0.9492424130439758), 0), ((2.180703639984131, 0.9492424130439758), 1), ((2.180703639984131, 0.9492424130439758), 2), ((2.1824355125427246, 0.9491737484931946), 3), ((2.181138515472412, 0.9472946524620056), 4), ((2.181138515472412, 0.9472946524620056), 5), ((2.181138515472412, 0.9472946524620056), 6), ((2.180703639984131, 0.9492424130439758), 7), ((0.8420583009719849, -1.0975843667984009), 8), ((2.180703639984131, 0.9492424130439758), 9)], 
                            # no outlier
                            [((0.3092443346977234, 2.4734885692596436), 0), ((0.3092443346977234, 2.4734885692596436), 1), ((0.3092443346977234, 2.4734885692596436), 2), ((0.3092443346977234, 2.4734885692596436), 3), ((0.3102819323539734, 2.473497152328491), 4), ((0.3102819323539734, 2.473497152328491), 5), ((0.3102819323539734, 2.473497152328491), 6), ((0.3102819323539734, 2.473497152328491), 7), ((0.3102819323539734, 2.473497152328491), 8), ((0.3102819323539734, 2.473497152328491), 9)], 
                            # frame 3, 5  are outliers and should be in next path, 6, 7, 8 should be in next next path
                            [((1.884087324142456, 2.394117593765259), 0), ((1.8822659254074097, 2.396878480911255), 1), ((1.8836601972579956, 2.395963191986084), 2), ((1.809637427330017, 2.6306309700012207), 3), ((1.8851573467254639, 2.3894946575164795), 4), ((1.8163647651672363, 2.6299242973327637), 5), ((1.207719326019287, 2.4844048023223877), 6), ((1.206670880317688, 2.4818367958068848), 7), ((1.2077878713607788, 2.480992078781128), 8), ((1.8853716850280762, 2.388568639755249), 9)], 
                            # frame 3 is outlier and should be in previous path
                            [((1.8027757406234741, 2.636873245239258), 0), ((1.8026002645492554, 2.637660503387451), 1), ((1.8047764301300049, 2.6376891136169434), 2), ((1.883545994758606, 2.3913393020629883), 3), ((1.8161851167678833, 2.6307153701782227), 4), ((1.2087836265563965, 2.4861178398132324), 5), ((1.8196405172348022, 2.6299664974212646), 6), ((1.8169171810150146, 2.6323108673095703), 7), ((1.886555790901184, 2.3885748386383057), 8), ((1.2525510787963867, 2.6443302631378174), 9)],
                            [((1.2131409645080566, 2.4878573417663574), 0), ((1.207651138305664, 2.4878101348876953), 1), ((1.2065365314483643, 2.4886510372161865), 2), ((1.2076852321624756, 2.4861085414886475), 3), ((1.208766222000122, 2.486968994140625), 4), ((2.2445361614227295, -0.6283740997314453), 5), ((2.2445361614227295, -0.6283740997314453), 6), ((2.07244873046875, -0.8683906197547913), 7), ((2.066617012023926, -0.8604583144187927), 8), ((1.2078220844268799, 2.479282855987549), 9)],
                            [((2.0737550258636475, -0.8611465096473694), 0), ((2.0761349201202393, -0.8613759875297546), 1), ((2.2457685470581055, -0.635342538356781), 2), ((2.2457685470581055, -0.635342538356781), 3), ((2.2457685470581055, -0.635342538356781), 4), ((1.886555790901184, 2.3885748386383057), 5), ((2.07244873046875, -0.8683906197547913), 6), ((2.2468504905700684, -0.628579318523407), 7), ((2.2468504905700684, -0.628579318523407), 8), ((2.2468504905700684, -0.628579318523407), 9)], 
                            [((2.2457685470581055, -0.635342538356781), 0), ((2.2457685470581055, -0.635342538356781), 1), ((2.0737550258636475, -0.8611465096473694), 2), ((2.0761349201202393, -0.8613759875297546), 3), ((2.0761349201202393, -0.8613759875297546), 4), ((1.249333143234253, 2.646589517593384), 5), ((1.886555790901184, 2.3885748386383057), 6), ((1.8882784843444824, 2.391364574432373), 7), ((1.8216437101364136, 2.6307859420776367), 8), ((2.07244873046875, -0.8683906197547913), 9)], 
                            [((1.2543866634368896, 2.651254653930664), 0), ((1.249387264251709, 2.6450552940368652), 1), ((1.2482348680496216, 2.6481077671051025), 2), ((1.2482885122299194, 2.6465752124786377), 3), ((1.2482885122299194, 2.6465752124786377), 4), ((2.074831247329712, -0.8686208724975586), 5), ((1.249387264251709, 2.6450552940368652), 6), ((1.2533714771270752, 2.65047550201416), 7), ((1.2504324913024902, 2.6450695991516113), 8), ((1.8205517530441284, 2.6307718753814697), 9)], 
                            [((2.1150639057159424, -1.3949592113494873), 0), ((2.117642879486084, -1.3952505588531494), 1), ((2.1991593837738037, 2.3174614906311035), 2), ((2.2016279697418213, 2.3174688816070557), 3), ((2.1143064498901367, -1.4078139066696167), 4), ((2.119474172592163, -1.4083998203277588), 5), ((2.119474172592163, -1.4083998203277588), 6), ((2.1200859546661377, -1.4127938747406006), 7), ((2.1168899536132812, -1.4081069231033325), 8), ((2.1200859546661377, -1.4127938747406006), 9)], 
                            [((2.1995015144348145, 2.312563419342041), 0), ((2.1988444328308105, 2.318438768386841), 1), ((2.1150639057159424, -1.3949592113494873), 2), ((2.1156718730926514, -1.399336338043213), 3), ((2.2004497051239014, 2.309619665145874), 4), ((2.2004497051239014, 2.309619665145874), 5), ((2.201399326324463, 2.306670665740967), 6), ((2.2004497051239014, 2.309619665145874), 7), ((2.2004497051239014, 2.309619665145874), 8), ((2.2001333236694336, 2.3106014728546143), 9)], 
                            [((0.023588383570313454, -1.6749101877212524), 3), ((0.023588383570313454, -1.6749101877212524), 4), ((0.028446096926927567, -1.6755427122116089), 5), ((0.028446096926927567, -1.6755427122116089), 6), ((0.028446096926927567, -1.6755427122116089), 7), ((0.028446096926927567, -1.6755427122116089), 8), ((0.028446096926927567, -1.6755427122116089), 9)], 
                            [], [], [], [], [], [], []
                            ]

def test_initialize_multi_tracker():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 4

def test_first_global_track():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4
   
    multi_tracker.globally_match_tracks()
    
    assert len(multi_tracker.globally_tracked) == 1

def test_track_multiple_frames():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4

    multi_tracker.globally_match_tracks()
    assert len(multi_tracker.globally_tracked) == 1
    assert len(multi_tracker.global_detections) == 0
    
    # Simulate tracking in the next frame with the same detections
    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4

    multi_tracker.globally_match_tracks()
    assert len(multi_tracker.globally_tracked) == 1    
    assert len(multi_tracker.global_detections) == 0

def test_no_detections():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    # Simulate no detections in the first frame
    multi_tracker.track([], 8)  # Camera 8
    multi_tracker.track([], 9)  # Camera 9
    multi_tracker.track([], 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 0

    with pytest.raises(ValueError, match="No global detections to match."):
        multi_tracker.globally_match_tracks()

def test_no_clusters():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    # Simulate only 2 detections from cam 8 (no cluster)
    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track([], 9)  # Camera 9
    multi_tracker.track([], 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 2

    with pytest.raises(ValueError, match="No clusters found."):
        multi_tracker.globally_match_tracks()

def test_handle_outiers():
    mapper = Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION)
    multi_tracker = MultiTracker(config.NUM_CAMERAS, config.FIRST_CAMERA, mapper, config.CLUSTER_EPSILON, config.CLUSTER_MIN_SAMPLES, config.MAX_GLOBAL_AGE, config.MAX_CLUSTER_DISTANCE, False)

    multi_tracker.processed_paths_by_cam[9] = dummy_batch_path_cam9
    multi_tracker.handle_outliers(config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES)

    for i in range(len(dummy_batch_path_cam9)):
        print(f"\n\nPATH {i} : {multi_tracker.processed_paths_by_cam[9][i]}")