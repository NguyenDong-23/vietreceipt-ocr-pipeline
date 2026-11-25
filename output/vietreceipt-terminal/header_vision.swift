import Foundation
import Vision

struct Crop: Decodable {
    let id: String
    let path: String
}

struct Reading: Encodable {
    let text: String
    let confidence: Double
}

let crops = try JSONDecoder().decode([Crop].self, from: FileHandle.standardInput.readDataToEndOfFile())
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
let supported = try request.supportedRecognitionLanguages()
guard let vietnamese = supported.first(where: { $0.hasPrefix("vi-") }) else {
    throw CocoaError(.featureUnsupported)
}
request.recognitionLanguages = [vietnamese, "en-US"]
request.usesLanguageCorrection = true
var readings = [String: Reading]()
for crop in crops {
    try VNImageRequestHandler(url: URL(fileURLWithPath: crop.path)).perform([request])
    let matches = (request.results ?? []).compactMap { $0.topCandidates(1).first }
    if !matches.isEmpty {
        readings[crop.id] = Reading(
            text: matches.map { $0.string }.joined(separator: " "),
            confidence: matches.map { Double($0.confidence) }.reduce(0, +) / Double(matches.count)
        )
    }
}
FileHandle.standardOutput.write(try JSONEncoder().encode(readings))
