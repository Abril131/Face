const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        video.srcObject = stream;
    })
    .catch(function(err) {
        console.error('Error accessing the camera: ', err);
    });

video.addEventListener('play', function() {
    const faceCascade = new cv.CascadeClassifier();
    faceCascade.load(cv.HAAR_FRONTALFACE_DEFAULT);

    setInterval(function() {
        context.drawImage(video, 0, 0, 640, 480);
        const frame = context.getImageData(0, 0, 640, 480);
        const src = cv.matFromImageData(frame);
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

        const faces = new cv.RectVector();
        const size = new cv.Size(0, 0);
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, size, size);

        for (let i = 0; i < faces.size(); ++i) {
            const face = faces.get(i);
            const point1 = new cv.Point(face.x, face.y);
            const point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
        }

        cv.imshow('canvas', src);
        src.delete();
        gray.delete();
        faces.delete();
    }, 100);
});

// Load OpenCV
cv.onRuntimeInitialized = () => {
    console.log('OpenCV loaded');
};
