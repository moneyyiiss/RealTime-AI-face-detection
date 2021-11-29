"use strict";
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var box_1 = require("./box");
var util_1 = require("./util");
var UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;
var LANDMARKS_COUNT = 468;
var MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [1, 168];
var BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [3, 2];
// The Pipeline coordinates between the bounding box and skeleton models.
var Pipeline = /** @class */ (function () {
    function Pipeline(boundingBoxDetector, meshDetector, meshWidth, meshHeight, maxContinuousChecks, maxFaces) {
        // An array of facial bounding boxes.
        this.regionsOfInterest = [];
        this.runsWithoutFaceDetector = 0;
        this.boundingBoxDetector = boundingBoxDetector;
        this.meshDetector = meshDetector;
        this.meshWidth = meshWidth;
        this.meshHeight = meshHeight;
        this.maxContinuousChecks = maxContinuousChecks;
        this.maxFaces = maxFaces;
    }
    Pipeline.prototype.transformRawCoords = function (rawCoords, box, angle, rotationMatrix) {
        var _this = this;
        var boxSize = box_1.getBoxSize({ startPoint: box.startPoint, endPoint: box.endPoint });
        var scaleFactor = [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];
        var coordsScaled = rawCoords.map(function (coord) {
            return [
                scaleFactor[0] * (coord[0] - _this.meshWidth / 2),
                scaleFactor[1] * (coord[1] - _this.meshHeight / 2), coord[2]
            ];
        });
        var coordsRotationMatrix = util_1.buildRotationMatrix(angle, [0, 0]);
        var coordsRotated = coordsScaled.map(function (coord) {
            var rotated = util_1.rotatePoint(coord, coordsRotationMatrix);
            return rotated.concat([coord[2]]);
        });
        var inverseRotationMatrix = util_1.invertTransformMatrix(rotationMatrix);
        var boxCenter = box_1.getBoxCenter({ startPoint: box.startPoint, endPoint: box.endPoint }).concat([
            1
        ]);
        var originalBoxCenter = [
            util_1.dot(boxCenter, inverseRotationMatrix[0]),
            util_1.dot(boxCenter, inverseRotationMatrix[1])
        ];
        return coordsRotated.map(function (coord) { return ([
            coord[0] + originalBoxCenter[0],
            coord[1] + originalBoxCenter[1], coord[2]
        ]); });
    };
    /**
     * Returns an array of predictions for each face in the input.
     *
     * @param input - tensor of shape [1, H, W, 3].
     */
    Pipeline.prototype.predict = function (input) {
        return __awaiter(this, void 0, void 0, function () {
            var returnTensors, annotateFace, _a, boxes, scaleFactor_1, scaledBoxes;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        if (!this.shouldUpdateRegionsOfInterest()) return [3 /*break*/, 2];
                        returnTensors = false;
                        annotateFace = true;
                        return [4 /*yield*/, this.boundingBoxDetector.getBoundingBoxes(input, returnTensors, annotateFace)];
                    case 1:
                        _a = _b.sent(), boxes = _a.boxes, scaleFactor_1 = _a.scaleFactor;
                        if (boxes.length === 0) {
                            this.regionsOfInterest = [];
                            return [2 /*return*/, null];
                        }
                        scaledBoxes = boxes.map(function (prediction) {
                            var predictionBoxCPU = {
                                startPoint: prediction.box.startPoint.squeeze().arraySync(),
                                endPoint: prediction.box.endPoint.squeeze().arraySync()
                            };
                            var scaledBox = box_1.scaleBoxCoordinates(predictionBoxCPU, scaleFactor_1);
                            var enlargedBox = box_1.enlargeBox(scaledBox);
                            return __assign({}, enlargedBox, { landmarks: prediction.landmarks.arraySync() });
                        });
                        boxes.forEach(function (box) {
                            if (box != null && box.startPoint != null) {
                                box.startEndTensor.dispose();
                                box.startPoint.dispose();
                                box.endPoint.dispose();
                            }
                        });
                        this.updateRegionsOfInterest(scaledBoxes);
                        this.runsWithoutFaceDetector = 0;
                        return [3 /*break*/, 3];
                    case 2:
                        this.runsWithoutFaceDetector++;
                        _b.label = 3;
                    case 3: return [2 /*return*/, tf.tidy(function () {
                            return _this.regionsOfInterest.map(function (box, i) {
                                var angle;
                                // The facial bounding box landmarks could come either from blazeface
                                // (if we are using a fresh box), or from the mesh model (if we are
                                // reusing an old box).
                                var boxLandmarksFromMeshModel = box.landmarks.length === LANDMARKS_COUNT;
                                if (boxLandmarksFromMeshModel) {
                                    var indexOfNose = MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[0], indexOfForehead = MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[1];
                                    angle = util_1.computeRotation(box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
                                }
                                else {
                                    var indexOfNose = BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[0], indexOfForehead = BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[1];
                                    angle = util_1.computeRotation(box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
                                }
                                var faceCenter = box_1.getBoxCenter({ startPoint: box.startPoint, endPoint: box.endPoint });
                                var faceCenterNormalized = [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];
                                var rotatedImage = tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);
                                var rotationMatrix = util_1.buildRotationMatrix(-angle, faceCenter);
                                var boxCPU = { startPoint: box.startPoint, endPoint: box.endPoint };
                                var face = box_1.cutBoxFromImageAndResize(boxCPU, rotatedImage, [
                                    _this.meshHeight, _this.meshWidth
                                ]).div(255);
                                // The first returned tensor represents facial contours, which are
                                // included in the coordinates.
                                var _a = _this.meshDetector.predict(face), flag = _a[1], coords = _a[2];
                                var coordsReshaped = tf.reshape(coords, [-1, 3]);
                                var rawCoords = coordsReshaped.arraySync();
                                var transformedCoordsData = _this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
                                var transformedCoords = tf.tensor2d(transformedCoordsData);
                                var landmarksBox = _this.calculateLandmarksBoundingBox(transformedCoordsData);
                                _this.regionsOfInterest[i] = __assign({}, landmarksBox, { landmarks: transformedCoords.arraySync() });
                                var prediction = {
                                    coords: coordsReshaped,
                                    scaledCoords: transformedCoords,
                                    box: landmarksBox,
                                    flag: flag.squeeze()
                                };
                                return prediction;
                            });
                        })];
                }
            });
        });
    };
    // Updates regions of interest if the intersection over union between
    // the incoming and previous regions falls below a threshold.
    Pipeline.prototype.updateRegionsOfInterest = function (boxes) {
        for (var i = 0; i < boxes.length; i++) {
            var box = boxes[i];
            var previousBox = this.regionsOfInterest[i];
            var iou = 0;
            if (previousBox && previousBox.startPoint) {
                var _a = box.startPoint, boxStartX = _a[0], boxStartY = _a[1];
                var _b = box.endPoint, boxEndX = _b[0], boxEndY = _b[1];
                var _c = previousBox.startPoint, previousBoxStartX = _c[0], previousBoxStartY = _c[1];
                var _d = previousBox.endPoint, previousBoxEndX = _d[0], previousBoxEndY = _d[1];
                var xStartMax = Math.max(boxStartX, previousBoxStartX);
                var yStartMax = Math.max(boxStartY, previousBoxStartY);
                var xEndMin = Math.min(boxEndX, previousBoxEndX);
                var yEndMin = Math.min(boxEndY, previousBoxEndY);
                var intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
                var boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
                var previousBoxArea = (previousBoxEndX - previousBoxStartX) *
                    (previousBoxEndY - boxStartY);
                iou = intersection / (boxArea + previousBoxArea - intersection);
            }
            if (iou < UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
                this.regionsOfInterest[i] = box;
            }
        }
        this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
    };
    Pipeline.prototype.clearRegionOfInterest = function (index) {
        if (this.regionsOfInterest[index] != null) {
            this.regionsOfInterest = this.regionsOfInterest.slice(0, index).concat(this.regionsOfInterest.slice(index + 1));
        }
    };
    Pipeline.prototype.shouldUpdateRegionsOfInterest = function () {
        var roisCount = this.regionsOfInterest.length;
        var noROIs = roisCount === 0;
        if (this.maxFaces === 1 || noROIs) {
            return noROIs;
        }
        return roisCount !== this.maxFaces &&
            this.runsWithoutFaceDetector >= this.maxContinuousChecks;
    };
    Pipeline.prototype.calculateLandmarksBoundingBox = function (landmarks) {
        var xs = landmarks.map(function (d) { return d[0]; });
        var ys = landmarks.map(function (d) { return d[1]; });
        var startPoint = [Math.min.apply(Math, xs), Math.min.apply(Math, ys)];
        var endPoint = [Math.max.apply(Math, xs), Math.max.apply(Math, ys)];
        var box = { startPoint: startPoint, endPoint: endPoint };
        return box_1.enlargeBox({ startPoint: box.startPoint, endPoint: box.endPoint });
    };
    return Pipeline;
}());
exports.Pipeline = Pipeline;
//# sourceMappingURL=pipeline.js.map