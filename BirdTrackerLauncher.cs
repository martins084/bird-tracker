using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class BirdTrackerLauncher
{
    [STAThread]
    private static void Main(string[] args)
    {
        string folder = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(folder, "gui.py");
        string python = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            @"Python\pythoncore-3.14-64\pythonw.exe");
        try
        {
            if (!File.Exists(script))
                throw new FileNotFoundException("Keep BirdTracker.exe in the same folder as gui.py and tracker.py.");
            if (!File.Exists(python))
                throw new FileNotFoundException("The Python installation used by Bird tracker could not be found.");
            bool selfTest = args.Length == 1 && args[0] == "--self-test";
            Process process = Process.Start(new ProcessStartInfo
            {
                FileName = python,
                Arguments = selfTest
                    ? "-c \"import gui; app=gui.App(); app.after(1000, app.destroy); app.mainloop()\""
                    : "\"" + script + "\"",
                WorkingDirectory = folder,
                UseShellExecute = false,
                CreateNoWindow = true
            });
            if (selfTest)
            {
                process.WaitForExit();
                Environment.ExitCode = process.ExitCode;
            }
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Bird tracker", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
