package com.screenscrub.vision

import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Classifies samples/{screen,document,other}/ and prints a confusion matrix plus per-image
 * features (CSV) for threshold tuning. Run: ./gradlew :vision:tune -Psamples=/path
 */
fun main(args: Array<String>) {
    Synthetic.load()
    val root = File(args.firstOrNull() ?: "samples")
    val scrub = ScreenScrub()
    // "document" photos, if present, are negatives: the app only cleans screens.
    val dirs = listOf("screen" to ImageClass.SCREEN, "other" to ImageClass.OTHER, "document" to ImageClass.OTHER)
    val classes = listOf("screen" to ImageClass.SCREEN, "other" to ImageClass.OTHER)
    val matrix = Array(2) { IntArray(2) }
    val csv = StringBuilder("file,expected,predicted,screenScore," + Features().asMap().keys.joinToString(",") + "\n")
    val cleanedDir = File(root, "cleaned")
    cleanedDir.deleteRecursively()
    var total = 0
    val debugDir = File(root, "debug")
    debugDir.deleteRecursively()
    for (pair in dirs) {
        val i = classes.indexOfFirst { it.second == pair.second }
        val dir = File(root, pair.first)
        val files = dir.listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png") }?.sorted() ?: emptyList()
        for (f in files) {
            val img = Imgcodecs.imread(f.absolutePath, Imgcodecs.IMREAD_COLOR)
            if (img.empty()) { println("skip unreadable ${f.name}"); continue }
            val t0 = System.nanoTime()
            val c = scrub.classify(img)
            val ms = (System.nanoTime() - t0) / 1_000_000
            val j = classes.indexOfFirst { it.second == c.imageClass }
            if (pair.first == "screen" && c.imageClass == ImageClass.SCREEN) {
                // Cleaned output for every detected screen, for a visual check.
                cleanedDir.mkdirs()
                scrub.clean(img, c, OutputStyle.AUTO)?.let { File(cleanedDir, "${f.nameWithoutExtension}_clean.${it.extension}").writeBytes(it.bytes) }
            }
            if (i != j) {
                // Overlay of every mistake, for a quick visual check of the detected quad.
                debugDir.mkdirs()
                val overlay = scrub.debugOverlay(img, c, 800)
                Imgcodecs.imwrite(File(debugDir, "${pair.first}_as_${c.imageClass.name.lowercase()}_${f.nameWithoutExtension.take(40)}.jpg").absolutePath, overlay)
                overlay.release()
            }
            img.release()
            matrix[i][j]++
            total++
            val mark = if (i == j) " " else "✗"
            println("$mark ${pair.first}/${f.name}: ${c.imageClass} s=%.2f (%d ms)".format(c.screenScore, ms))
            csv.append("${f.name},${pair.second},${c.imageClass},%.3f,".format(c.screenScore))
            csv.append(c.features.asMap().values.joinToString(",") { "%.4f".format(it) }).append('\n')
        }
    }
    if (total == 0) {
        println("No images found under ${root.absolutePath}/{screen,document,other}/")
        return
    }
    println("\nConfusion matrix (rows = expected, columns = predicted)")
    println("%-10s %8s %8s %8s".format("", "SCREEN", "OTHER", "recall"))
    for (i in 0..1) {
        val n = matrix[i].sum()
        val recall = if (n == 0) "-" else "%.0f%%".format(100.0 * matrix[i][i] / n)
        println("%-10s %8d %8d %8s".format(classes[i].second, matrix[i][0], matrix[i][1], recall))
    }
    val others = matrix[1].sum()
    if (others > 0) println("False positives on non-screen photos: %.1f%% (target <= 1%%)".format(100.0 * matrix[1][0] / others))
    val out = File(root, "features.csv")
    out.writeText(csv.toString())
    println("Features written to ${out.absolutePath}")
}
