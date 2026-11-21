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
    val classes = listOf("screen" to ImageClass.SCREEN, "document" to ImageClass.DOCUMENT, "other" to ImageClass.OTHER)
    val matrix = Array(3) { IntArray(3) }
    val csv = StringBuilder("file,expected,predicted,screenScore,docScore," + Features().asMap().keys.joinToString(",") + "\n")
    var total = 0
    for ((i, pair) in classes.withIndex()) {
        val dir = File(root, pair.first)
        val files = dir.listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png") }?.sorted() ?: emptyList()
        for (f in files) {
            val img = Imgcodecs.imread(f.absolutePath, Imgcodecs.IMREAD_COLOR)
            if (img.empty()) { println("skip unreadable ${f.name}"); continue }
            val t0 = System.nanoTime()
            val c = scrub.classify(img)
            val ms = (System.nanoTime() - t0) / 1_000_000
            img.release()
            val j = classes.indexOfFirst { it.second == c.imageClass }
            matrix[i][j]++
            total++
            val mark = if (i == j) " " else "✗"
            println("$mark ${pair.first}/${f.name}: ${c.imageClass} s=%.2f d=%.2f (%d ms)".format(c.screenScore, c.documentScore, ms))
            csv.append("${f.name},${pair.second},${c.imageClass},%.3f,%.3f,".format(c.screenScore, c.documentScore))
            csv.append(c.features.asMap().values.joinToString(",") { "%.4f".format(it) }).append('\n')
        }
    }
    if (total == 0) {
        println("No images found under ${root.absolutePath}/{screen,document,other}/")
        return
    }
    println("\nConfusion matrix (rows = expected, columns = predicted)")
    println("%-10s %8s %8s %8s %8s".format("", "SCREEN", "DOCUMENT", "OTHER", "recall"))
    for (i in 0..2) {
        val n = matrix[i].sum()
        val recall = if (n == 0) "-" else "%.0f%%".format(100.0 * matrix[i][i] / n)
        println("%-10s %8d %8d %8d %8s".format(classes[i].second, matrix[i][0], matrix[i][1], matrix[i][2], recall))
    }
    val others = matrix[2].sum()
    if (others > 0) println("False positives on normal photos: %.1f%% (target ≤ 1%%)".format(100.0 * (matrix[2][0] + matrix[2][1]) / others))
    val out = File(root, "features.csv")
    out.writeText(csv.toString())
    println("Features written to ${out.absolutePath}")
}
