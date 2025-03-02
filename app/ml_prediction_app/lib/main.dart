import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:csv/csv.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ML Prediction App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final _formKey = GlobalKey<FormState>();
  final Map<String, TextEditingController> _controllers = {};
  Map<String, dynamic> _predictionData = {};

  @override
  void initState() {
    super.initState();
    List<String> fields = [
      'Active Component Density',
      'Active Component Formation Energy',
      'Active Component Content',
      'Promoter Density',
      'Promoter Formation Energy',
      'Promoter Content',
      'Support A Density',
      'Support A Formation Energy',
      'Support A Content',
      'Support B Density',
      'Support B Formation Energy',
      'Preparation Scalability',
      'Preparation Cost',
      'Calcination Temperature',
      'Calcination Time',
      'Reduction Temperature',
      'Reduction Pressure',
      'Reduction Time',
      'Process Temperature',
      'Process Pressure',
      'WHSV',
      'Content of Inert Components',
      'H2CO2 Ratio'
    ];
    for (var field in fields) {
      _controllers[field] = TextEditingController();
    }
  }

  Future<void> _getPredictionFromManualInput() async {
    // Collect the user input from form
    Map<String, dynamic> inputData = {};
    _controllers.forEach((key, controller) {
      inputData[key] = controller.text;
    });

    final response = await http.post(
      Uri.parse('http://localhost:5000/predict'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(inputData),
    );

    if (response.statusCode == 200) {
      var data = json.decode(response.body);
      setState(() {
        // Saving the response data
        _predictionData = {
          'CO2 Conversion Ratio': data['prediction'],
          'MSE': data['mse'],
          'R2': data['r2']
        };
      });
    } else {
      setState(() {
        _predictionData = {};
      });
    }
  }

  Future<void> _getPredictionFromCSV() async {
    FilePickerResult? result = await FilePicker.platform
        .pickFiles(type: FileType.custom, allowedExtensions: ['csv']);
    if (result != null) {
      File file = File(result.files.single.path!);
      String csvData = await file.readAsString();

      final response = await http.post(
        Uri.parse('http://localhost:5000/predict-csv'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'csv_data': csvData}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        setState(() {
          // Saving the response data with more descriptive names
          _predictionData = {
            'CO2 Conversion Ratio': data['prediction'],
            'MSE': data['mse'],
            'R2': data['r2']
          };
        });
      } else {
        setState(() {
          _predictionData = {};
        });
      }
    }
  }

// Function to save the results as a CSV file
// Function to save the results as a CSV file
  Future<void> _saveResultsToCSV() async {
    if (_predictionData.isNotEmpty) {
      // Prepare the CSV data
      List<List<dynamic>> rows = [];
      rows.add(["Index", "Metric", "Value"]); // Custom headers for CSV

      // Adding data to CSV with an index
      int index = 1;
      _predictionData.forEach((key, value) {
        rows.add(
            [index, key, value]); // Adding the index, metric name, and value
        index++;
      });

      // Get the device's storage directory
      Directory appDocDir = await getApplicationDocumentsDirectory();
      String filePath = '${appDocDir.path}/prediction_results.csv';

      // Create a CSV file and write the rows to it
      File file = File(filePath);
      String csv = const ListToCsvConverter().convert(rows);
      await file.writeAsString(csv);

      // Notify the user
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Results saved to $filePath')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('ML Prediction App'),
      ),
      backgroundColor: Colors.white, // Background color set to white
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20),
        child: Center(
          // Center content for smaller windows
          child: ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: 400, // Limit the width of the content
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Manual input form
                Form(
                  key: _formKey,
                  child: Column(
                    children: _controllers.keys.map((field) {
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 5.0),
                        child: TextFormField(
                          controller: _controllers[field],
                          decoration: InputDecoration(
                            labelText: field,
                            border:
                                OutlineInputBorder(), // Add borders for clarity
                            contentPadding:
                                EdgeInsets.symmetric(horizontal: 10),
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter a value';
                            }
                            return null;
                          },
                        ),
                      );
                    }).toList(),
                  ),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    if (_formKey.currentState?.validate() ?? false) {
                      _getPredictionFromManualInput();
                    }
                  },
                  child: Text('Predict from Input'),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _getPredictionFromCSV,
                  child: Text('Upload CSV and Predict'),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _saveResultsToCSV,
                  child: Text('Save Results to CSV'),
                ),
                SizedBox(height: 40),
                // Prediction results with custom labels
                if (_predictionData.isNotEmpty)
                  ..._predictionData.entries.map((entry) {
                    return Text('${entry.key}: ${entry.value}');
                  }),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
