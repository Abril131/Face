function updateFaceBoxes(face_locations, face_names) {
        $("#face_boxes").empty();
        for (var i = 0; i < face_locations.length; i++) {
            var face_location = face_locations[i];
            var face_name = face_names[i];
            var top = face_location[0];
            var right = face_location[1];
            var bottom = face_location[2];
            var left = face_location[3];
            var box_width = right - left;
            var box_height = bottom - top;
            var box_html = '<div class="face-box" style="top: '+top+'px; left: '+left+'px; width: '+box_width+'px; height: '+box_height+'px;">'+face_name+'</div>';
            $("#face_boxes").append(box_html);
        }
    }